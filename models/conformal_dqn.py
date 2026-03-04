import copy
import numpy as np
import torch
import torch.nn.functional as F
from .networks import DDQNSF


class ConformalDQN:
    def __init__(self, params):
        self.device = params.get("device")
        self.num_actions = params.get("num_actions")
        self.discount = params.get("discount")
        self.tau_update = params.get("tau")

        self.Q = DDQNSF(
            params.get("state_dim"),
            params.get("num_actions"),
            params.get("layer_size"),
            params.get("temperature", 2.0)
        ).to(self.device)
        self.Q_target = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=params.get("lr"))

        self.epsilon = params.get("epsilon_start")
        self.epsilon_end = params.get("epsilon_end")
        self.epsilon_decay = params.get("epsilon_decay")
        self.conformal_predictor = None

        self.phase_bins = params.get("phase_bins", (0.15, 0.85))
        self.entropy_weight = float(params.get("entropy_weight", 0.0))
        self.entropy_targets = params.get("entropy_targets", (0.6, 1.8, 1.0))
        self.prior_weight = float(params.get("prior_weight", 0.0))

        # 【核心修复】：提取专家克隆的权重！
        self.bc_weight = float(params.get("bc_weight", 5.0))

    def select_action(self, state, eval_mode=False):
        if not eval_mode:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if np.random.rand() < self.epsilon:
                return int(np.random.randint(self.num_actions))

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, action_log_probs, _ = self.Q(state_tensor)

            if self.conformal_predictor is None or not hasattr(self.conformal_predictor, 'threshold_global'):
                return int(q_values.argmax(1))

            action_probs = action_log_probs.exp().squeeze(0)

            threshold = self.conformal_predictor.get_threshold(state)
            safe_mask = action_probs >= threshold

            if safe_mask.sum().item() == 0:
                max_idx = int(action_probs.argmax(0))
                safe_mask.__setitem__(max_idx, True)

            confident_mask = safe_mask.float().unsqueeze(0)
            confident_q_values = confident_mask * q_values + (1.0 - confident_mask) * -1e8
            return int(confident_q_values.argmax(1))

    def train(self, online_buffer, offline_bc_buffer, batch_size):
        state, action, reward, not_done, next_state = online_buffer.sample(batch_size)

        with torch.no_grad():
            next_q, _, _ = self.Q(next_state)
            next_action = next_q.argmax(1, keepdim=True)
            target_q, _, _ = self.Q_target(next_state)
            target_q_val = target_q.gather(1, next_action).squeeze(1)
            expected_q = reward.squeeze(1) + not_done.squeeze(1) * self.discount * target_q_val

        current_q, _, _ = self.Q(state)
        current_q = current_q.gather(1, action).squeeze(1)
        q_loss = F.huber_loss(current_q, expected_q)

        bc_state, bc_action, _, _, _ = offline_bc_buffer.sample(batch_size)
        _, bc_log_probs, bc_logits = self.Q(bc_state)

        nll_loss = F.nll_loss(bc_log_probs, bc_action.squeeze(1))

        entropy_loss = 0.0
        if self.entropy_weight > 0.0:
            probs = bc_log_probs.exp()
            entropy = -(probs * bc_log_probs).sum(dim=1)

            s_ratio = bc_state[:, 0]
            target_start, target_cruise, target_brake = self.entropy_targets
            target_entropy = torch.where(
                s_ratio < self.phase_bins[0],
                torch.full_like(s_ratio, float(target_start)),
                torch.where(
                    s_ratio < self.phase_bins[1],
                    torch.full_like(s_ratio, float(target_cruise)),
                    torch.full_like(s_ratio, float(target_brake))
                )
            )
            entropy_loss = (entropy - target_entropy).pow(2).mean()

        prior_loss = 0.0
        if self.prior_weight > 0.0:
            s_ratio = bc_state[:, 0]
            phase_ids = torch.zeros_like(s_ratio, dtype=torch.long)
            phase_ids = torch.where(s_ratio < self.phase_bins[0], phase_ids, torch.ones_like(phase_ids))
            phase_ids = torch.where(s_ratio < self.phase_bins[1], phase_ids, torch.full_like(phase_ids, 2))

            action_ids = bc_action.squeeze(1)
            num_actions = self.num_actions
            alpha = 0.5
            priors = torch.full((3, num_actions), 1.0 / num_actions, device=bc_log_probs.device)

            for pid in (0, 1, 2):
                idx = (phase_ids == pid).nonzero(as_tuple=False).squeeze(1)
                if idx.numel() > 0:
                    counts = torch.bincount(action_ids.index_select(0, idx), minlength=num_actions).float()
                    priors[pid] = (counts + alpha) / (counts.sum() + alpha * num_actions)

            target_probs = priors.index_select(0, phase_ids)
            prior_loss = F.kl_div(bc_log_probs, target_probs, reduction="batchmean")

        # 【核心修复】：用 self.bc_weight 镇压 Q_loss 的野蛮生长！
        loss = q_loss + self.bc_weight * nll_loss + bc_logits.pow(
            2).mean() * 0.01 + self.entropy_weight * entropy_loss + self.prior_weight * prior_loss

        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止 Q 爆炸导致梯度失控
        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), max_norm=10.0)
        self.optimizer.step()

        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau_update * param.data + (1.0 - self.tau_update) * target_param.data)

        return q_loss.item(), nll_loss.item()