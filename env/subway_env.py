import numpy as np
import math
import gym
from gym import spaces
from gym.utils import seeding
from . import TrainAndRoadCharacter as trc
from . import trainRunningModel as trm


class TrainLine(gym.Env):
    def __init__(self, time, action_step=0.2, action_min=-0.8, action_max=0.8, num_actions=None, time_limit=None):

        self.startPoint = float(trc.SLStartPoint.__getitem__(0))
        self.endPoint = float(trc.SLStartPoint.__getitem__(len(trc.SLStartPoint) - 1))
        self.S = self.endPoint - self.startPoint
        self.max_speed = 80.0 / 3.6
        self.T = float(time)
        if self.T <= 0.0: self.T = 110.0
        self.dt = 0.2
        if time_limit is None:
            self.time_limit = self.T
        else:
            self.time_limit = float(time_limit)
            if self.time_limit <= 0.0:
                self.time_limit = self.T

        self.time_lower_ratio = 0.8
        self.time_upper_ratio = 1.2
        self.time_lower = self.T * self.time_lower_ratio
        self.time_upper = self.T * self.time_upper_ratio
        self.time_hard_cap_ratio = 3.0
        self.time_hard_cap = self.T * self.time_hard_cap_ratio
        self.time_hard_cap_penalty = 800.0
        self.max_steps = int(self.time_hard_cap / self.dt) + 1





        self.action_step = float(action_step)
        self.action_min = float(action_min)
        self.action_max = float(action_max)
        if num_actions is None:
            num_actions = int(round((self.action_max - self.action_min) / self.action_step)) + 1
        self.n_actions = int(num_actions)
        self.center_idx = (self.n_actions - 1) / 2.0


        # 【修改点 3】：物理环境支持 6 维输出（加入坡度感知）
        self.n_features = 6
        self.max_grad = 30.0
        self.low = np.array(list((0.0, 0.0, 0.0, -1.0, 0.0, -1.0)), dtype=np.float32)
        self.high = np.array(list((1.0, 1.0, 2.0, 1.0, 1.0, 1.0)), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_actions)

        self.observation_space = spaces.Box(self.low, self.high)

        self.seed()
        self.done = False
        self.filterFactor = 0.88




        # 训练稳定性与舒适性相关惩罚参数
        self.overspeed_hard_penalty = 18.0
        self.overspeed_extra_penalty = 140.0
        self.overspeed_soft_penalty = 22.0
        self.safe_limit_ratio = 0.9
        self.hard_limit_ratio = 0.95
        self.smooth_action_penalty = 20

        self.smooth_speed_penalty = 40
        self.cruise_smooth_speed_penalty = 50






        self.approach_ratio = 0.05

        self.approach_dist = max(1.0, self.S * self.approach_ratio)

        self.approach_penalty = 35.0
        self.approach_no_reaccel_penalty = 180.0



        self.downhill_ratio_scale = 0.005
        self.downhill_min_ratio = 0.985

        # 前瞻坡度与分区间参数（按限速区段）
        self.lookahead_dist = 120.0
        self.lookahead_ease_penalty = 8.0

        self.segment_count = max(1, len(trc.SLStartPoint))
        self.segment_cruise_penalty_scale = [1.0] * self.segment_count
        self.segment_smooth_speed_scale = [1.0] * self.segment_count
        self.segment_downhill_ratio_scale = [self.downhill_ratio_scale] * self.segment_count
        self.segment_downhill_min_ratio = [self.downhill_min_ratio] * self.segment_count
        self.segment_lookahead_scale = [1.0] * self.segment_count
        self.segment_uphill_boost_scale = [1.0] * self.segment_count

        # 第三、四区间上坡速度波动大：加大巡航低速惩罚与速度平滑惩罚，并增强前瞻
        for seg_idx in (2, 3):
            if seg_idx < self.segment_count:
                self.segment_cruise_penalty_scale[seg_idx] = 1.7
                self.segment_smooth_speed_scale[seg_idx] = 1.75
                self.segment_lookahead_scale[seg_idx] = 1.6
                self.segment_uphill_boost_scale[seg_idx] = 1.25

        # 第五区间巡航低速仍偏多：单独加强
        seg_idx = 4
        if seg_idx < self.segment_count:
            self.segment_cruise_penalty_scale[seg_idx] = 1.8
            self.segment_smooth_speed_scale[seg_idx] = 1.75
            self.segment_lookahead_scale[seg_idx] = 1.7
            self.segment_uphill_boost_scale[seg_idx] = 1.35

        # 第十三区间上坡掉速明显：追加增强
        seg_idx = 12
        if seg_idx < self.segment_count:
            self.segment_cruise_penalty_scale[seg_idx] = 1.7
            self.segment_smooth_speed_scale[seg_idx] = 1.7
            self.segment_lookahead_scale[seg_idx] = 1.6
            self.segment_uphill_boost_scale[seg_idx] = 1.35

        # 多个区间下坡掉速过快：大幅放松下坡限速收缩
        for seg_idx in (0, 3, 9, 10, 12):
            if seg_idx < self.segment_count:
                self.segment_downhill_ratio_scale[seg_idx] = 0.002
                self.segment_downhill_min_ratio[seg_idx] = 0.993













        self.stop_hold_dist = 50.0
        self.stop_hold_penalty = 80.0

        self.stop_speed_tol = 0.15
        self.stop_speed_penalty = 320.0
        self.stop_speed_bonus = 20.0

        self.mid_stop_dist = 120.0
        self.mid_stop_penalty = 260.0
        self.mid_stop_push = 0.2
        self.start_stop_steps = 1
        self.start_speed_penalty = 200.0



        self.pos_reward_scale = 0.6
        self.time_penalty = 0.4
        self.late_penalty = 260.0
        self.on_time_bonus = 300.0
        self.on_time_window = 0.05
        self.pace_lag_penalty = 14.0
        self.pace_margin = 0.02

        self.early_finish_penalty = 260.0
        self.late_finish_penalty = 260.0

        self.cruise_low_ratio = 0.7
        self.cruise_low_penalty = 420.0
        self.cruise_low_penalty_uphill_scale = 1.1
        self.cruise_low_penalty_downhill_scale = 1.25
        self.uphill_cruise_boost = 65.0
        self.uphill_boost_grad_scale = 0.08











        self.min_cruise_speed = 2.0
        self.min_cruise_penalty = 6.0

        self.early_brake_dist = 700.0
        self.early_brake_penalty = 8.0








        self.reset()




    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return list((seed,))

    def _get_segment_idx(self, pos):
        idx = 0
        for i in range(len(trc.SLStartPoint)):
            if pos >= trc.SLStartPoint[i]:
                idx = i
        return max(0, min(idx, len(trc.SLStartPoint) - 1))

    def step(self, action):

        reward = 0.0
        prev_u = self.u
        prev_veo = self.veo
        du = (float(action) - self.center_idx) * self.action_step

        u = du * (1.0 - self.filterFactor) + self.u * self.filterFactor
        u = max(self.action_min, min(self.action_max, u))


        # 【修改点 4】：ATP 超速硬保护（下坡更严格）
        limit_before = trc.getRoadspeedLimit(self.pos) * self.hard_limit_ratio

        current_grad = trc.getRoadGradinet(self.pos)
        segment_idx_before = self._get_segment_idx(self.pos)
        seg_downhill_ratio_scale = self.segment_downhill_ratio_scale[segment_idx_before]
        seg_downhill_min_ratio = self.segment_downhill_min_ratio[segment_idx_before]
        grad_scale = 1.0
        if current_grad < 0.0:
            grad_scale = max(seg_downhill_min_ratio, 1.0 + current_grad * seg_downhill_ratio_scale)

        limit_before = limit_before * grad_scale
        if self.veo > limit_before:
            u = -1.0
            reward -= self.overspeed_hard_penalty
            overspeed_ratio = (self.veo - limit_before) / self.max_speed
            reward -= self.overspeed_extra_penalty * (overspeed_ratio ** 2)
            action = 0



        # 保持状态输入范围一致
        u = max(self.action_min, min(self.action_max, u))


        remaining_before = self.endPoint - self.pos
        if remaining_before <= self.approach_dist and u > 0.0:
            reward -= self.approach_no_reaccel_penalty
            u = 0.0
        if remaining_before <= self.approach_dist and current_grad < 0.0 and u > -0.10:
            u = -0.10





        elif remaining_before > self.approach_dist and self.pos > self.startPoint + self.mid_stop_dist and self.veo <= self.stop_speed_tol:
            reward -= self.mid_stop_penalty
            u = max(u, self.mid_stop_push)


        # 平滑惩罚：抑制动作震荡（弱化），主要看速度变化
        reward -= self.smooth_action_penalty * abs(u - prev_u)



        self.u = u
        trainState = self.train.Step(self.u)


        pos_cha = float(trainState.get('S')) - self.pos
        self.pos = float(trainState.get('S'))

        # 【修改点 5】：绝对防倒车锁
        v_temp = float(trainState.get('v'))
        if v_temp < 0.0:
            self.veo = 0.0
            self.train.speed = 0.0
        else:
            self.veo = v_temp

        segment_idx = self._get_segment_idx(self.pos)
        seg_smooth_speed_penalty = self.smooth_speed_penalty * self.segment_smooth_speed_scale[segment_idx]
        seg_cruise_penalty_scale = self.segment_cruise_penalty_scale[segment_idx]
        seg_lookahead_dist = self.lookahead_dist * self.segment_lookahead_scale[segment_idx]
        seg_downhill_ratio_scale = self.segment_downhill_ratio_scale[segment_idx]
        seg_downhill_min_ratio = self.segment_downhill_min_ratio[segment_idx]

        # 速度平滑惩罚：抑制速度突变
        reward -= seg_smooth_speed_penalty * abs(self.veo - prev_veo)


        dE = float(trainState.get('P')) / trc.M

        self.EC += dE * self.dt

        if u > 0.0: reward -= dE * self.dt * 0.005
        reward += pos_cha * self.pos_reward_scale


        current_limit = trc.getRoadspeedLimit(self.pos) * self.hard_limit_ratio
        v_ratio = self.veo / self.max_speed

        limit_ratio = current_limit / self.max_speed
        safe_ratio = limit_ratio * self.safe_limit_ratio

        current_grad = trc.getRoadGradinet(self.pos)
        if current_grad < 0.0:
            grad_scale = max(seg_downhill_min_ratio, 1.0 + current_grad * seg_downhill_ratio_scale)
            safe_ratio *= grad_scale


        excess = v_ratio - max(0.0, safe_ratio)
        if excess > 0.0:
            reward -= self.overspeed_soft_penalty * (excess ** 2)

        remaining = self.endPoint - self.pos
        if remaining <= self.approach_dist:
            remaining = max(0.0, remaining)
            target_ratio = safe_ratio * (remaining / self.approach_dist)
            approach_excess = v_ratio - max(0.0, target_ratio)
            if approach_excess > 0.0:
                reward -= self.approach_penalty * (approach_excess ** 2)

        if remaining > self.stop_hold_dist and self.veo <= self.stop_speed_tol:
            reward -= self.stop_hold_penalty
        elif remaining <= self.stop_hold_dist and self.veo <= self.stop_speed_tol and u > 0.0:
            reward -= self.stop_hold_penalty * (u ** 2)

        if remaining > self.approach_dist and self.veo < self.min_cruise_speed:
            low_ratio = (self.min_cruise_speed - self.veo) / self.min_cruise_speed
            reward -= self.min_cruise_penalty * (low_ratio ** 2)

        if remaining > self.approach_dist:
            cruise_limit = current_limit * self.cruise_low_ratio
            cruise_penalty = self.cruise_low_penalty * seg_cruise_penalty_scale
            if current_grad > 0.0:
                cruise_penalty *= self.cruise_low_penalty_uphill_scale
            elif current_grad < 0.0:
                cruise_penalty *= self.cruise_low_penalty_downhill_scale

            # 巡航阶段速度平滑惩罚：抑制毛刺
            reward -= self.cruise_smooth_speed_penalty * (abs(self.veo - prev_veo) ** 1.4)


            if self.veo < cruise_limit:
                cruise_ratio = (cruise_limit - self.veo) / max(0.1, cruise_limit)
                reward -= cruise_penalty * (cruise_ratio ** 2)
                if current_grad > 0.0:
                    grad_factor = min(1.6, current_grad * self.uphill_boost_grad_scale + 0.2)
                    seg_boost = self.segment_uphill_boost_scale[segment_idx]
                    reward += self.uphill_cruise_boost * seg_boost * max(0.0, u) * (cruise_ratio ** 2) * grad_factor







        if remaining > self.early_brake_dist and u < 0.0:
            reward -= self.early_brake_penalty * (u ** 2)


        self.step1 += 1


        t_current = self.step1 * self.dt
        self.t_current = t_current

        reward -= self.time_penalty

        if t_current > self.T:
            late_ratio = (t_current - self.T) / self.T
            reward -= self.late_penalty * (late_ratio ** 2)

        if self.pos >= self.endPoint:

            self.done = True
            time_diff = abs(t_current - self.T)
            reward += 200.0 - time_diff * 2.0
            if time_diff <= self.T * self.on_time_window:
                reward += self.on_time_bonus
            if t_current < self.time_lower:
                reward -= self.early_finish_penalty
            elif t_current > self.time_upper:
                reward -= self.late_finish_penalty

            # 到站速度约束：硬性要求停稳

            if self.veo > self.stop_speed_tol:
                stop_ratio = self.veo / self.max_speed
                reward -= self.stop_speed_penalty * (stop_ratio ** 2)
                reward -= self.mid_stop_penalty
            else:
                reward += self.stop_speed_bonus

        elif t_current >= self.time_hard_cap:
            self.done = True
            self.done_reason = "time_hard_cap"
            reward -= self.time_hard_cap_penalty



        elif t_current >= self.time_upper:
            self.done = False
            reward -= self.late_finish_penalty



        else:
            self.done = False




        s_ratio = (self.pos - self.startPoint) / self.S
        v_ratio = self.veo / self.max_speed
        t_ratio = t_current / self.T

        limit_ratio = current_limit / self.max_speed
        grad_ratio = np.clip(trc.getRoadGradinet(self.pos) / self.max_grad, -1.0, 1.0)

        # 组装 6 维状态
        self.state = np.array(list((s_ratio, v_ratio, t_ratio, self.u, limit_ratio, grad_ratio)), dtype=np.float32)

        return self.state, self.EC, reward, self.done, action


    def reset(self):
        self.train = trm.Train_model(self.startPoint, 0.0, 0.0, self.dt)
        self.EC = 0.0
        self.pos = self.startPoint
        self.veo = 0.0
        self.u = 0.0
        self.step1 = 0
        self.t_current = 0.0
        self.done_reason = "reset"

        s_ratio = (self.pos - self.startPoint) / self.S

        current_limit = trc.getRoadspeedLimit(self.pos)
        limit_ratio = current_limit / self.max_speed
        grad_ratio = np.clip(trc.getRoadGradinet(self.pos) / self.max_grad, -1.0, 1.0)

        self.state = np.array(list((s_ratio, 0.0, 0.0, 0.0, limit_ratio, grad_ratio)), dtype=np.float32)
        return self.state
