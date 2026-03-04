import numpy as np
import torch


class ConformalPredictor:
    def __init__(self, policy, cal_buffer, confidence_level=0.85, phase_bins=(0.15, 0.85), min_samples=200, phase_confidence=(0.85, 0.95, 0.95)):
        self.policy = policy
        self.cal_buffer = cal_buffer
        self.confidence_level = confidence_level
        self.phase_bins = tuple(phase_bins)
        self.min_samples = int(min_samples)
        self.phase_confidence = tuple(phase_confidence)
        self.threshold_global = 0.15
        self.threshold_by_phase = dict()
        self.tau_global = 0.85
        self.tau_by_phase = dict()




    def _phase_id_from_s(self, s_ratio):
        if s_ratio < self.phase_bins[0]:
            return 0
        if s_ratio < self.phase_bins[1]:
            return 1
        return 2

    def _phase_ids_from_states(self, states_np):
        s_ratio = states_np[:, 0]
        phase_ids = np.zeros_like(s_ratio, dtype=np.int64)
        phase_ids[s_ratio >= self.phase_bins[0]] = 1
        phase_ids[s_ratio >= self.phase_bins[1]] = 2
        return phase_ids

    def get_threshold(self, state):
        if not self.threshold_by_phase:
            return self.threshold_global
        s_ratio = float(state[0])
        phase_id = self._phase_id_from_s(s_ratio)
        return float(self.threshold_by_phase.get(phase_id, self.threshold_global))




    def calibrate(self):
        states, actions = self.cal_buffer.get_all()
        with torch.no_grad():
            _, action_log_probs, _ = self.policy.Q(states)
            probs = action_log_probs.exp()

        # 获取专家采取的那个动作的具体概率
        expert_probs = probs.gather(1, actions).squeeze(1).cpu().numpy()

        # 目标：让专家动作在 >= confidence_level 的比例上被判为安全
        # 安全判定规则：action_prob >= threshold
        q_level = 1.0 - float(self.confidence_level)
        q_level = min(max(q_level, 0.0), 1.0)

        self.threshold_global = float(np.quantile(expert_probs, q_level, method="lower"))
        self.tau_global = 1.0 - self.threshold_global

        states_np = states.detach().cpu().numpy()
        phase_ids = self._phase_ids_from_states(states_np)
        self.threshold_by_phase = dict()
        self.tau_by_phase = dict()

        for phase_id in (0, 1, 2):
            idx = np.where(phase_ids == phase_id)[0]
            phase_conf = float(self.phase_confidence[phase_id]) if len(self.phase_confidence) > phase_id else float(self.confidence_level)
            phase_q = 1.0 - phase_conf
            phase_q = min(max(phase_q, 0.0), 1.0)
            if len(idx) >= self.min_samples:
                threshold_val = float(np.quantile(expert_probs[idx], phase_q, method="lower"))
            else:
                threshold_val = self.threshold_global
            self.threshold_by_phase[phase_id] = threshold_val
            self.tau_by_phase[phase_id] = 1.0 - threshold_val


        print(
            "CP 分阶段阈值校准完成！全局阈值 p*: "
            + str(round(self.threshold_global, 4))
            + " | 起步 p*: "
            + str(round(self.threshold_by_phase.get(0, self.threshold_global), 4))
            + " | 巡航 p*: "
            + str(round(self.threshold_by_phase.get(1, self.threshold_global), 4))
            + " | 进站 p*: "
            + str(round(self.threshold_by_phase.get(2, self.threshold_global), 4))
            + " | 对应 τ: "
            + str(round(self.tau_global, 4))
            + " | 分段置信度: "
            + str(tuple(self.phase_confidence))
        )




