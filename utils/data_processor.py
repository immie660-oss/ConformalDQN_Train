import os
import pandas as pd
import numpy as np
import torch
from .buffer import ReplayBuffer


def parse_csv_logs(data_dir, env_S, env_max_speed, env_T, env_max_grad=30.0, action_step=0.2, action_min=-0.8, action_max=0.8):


    expert_states = list()
    expert_actions = list()

    for file in os.listdir(data_dir):
        if not file.endswith('.csv'): continue
        file_path = os.path.join(data_dir, file)

        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig', encoding_errors='ignore', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(file_path, encoding='gbk', encoding_errors='ignore', on_bad_lines='skip')

        if '计算距离(m)' not in df.columns:
            continue

        vcol = None
        for cand in (
            '车速(km/h)', '车辆速度(km/h)', '车辆速度（km/h）', '速度(km/h)', '速度（km/h）',
            '车速数据(cm/s)', '车辆速度', '速度'
        ):
            if cand in df.columns:
                vcol = cand
                break
        if vcol is None:
            continue

        pos_vals = df['计算距离(m)'].to_numpy(dtype=float)
        vel_vals = df[vcol].to_numpy(dtype=float)
        if vel_vals.size == 0 or pos_vals.size == 0:
            continue

        vel_max = float(np.nanmax(vel_vals)) if np.isfinite(vel_vals).any() else 0.0
        if 'km/h' in vcol:
            vel_vals = vel_vals / 3.6
        elif 'cm/s' in vcol:
            vel_vals = vel_vals / 100.0
        elif vel_max > 200.0:
            vel_vals = vel_vals / 100.0
        elif vel_max > 60.0:
            vel_vals = vel_vals / 3.6

        n = int(min(len(pos_vals), len(vel_vals)))
        pos_vals = pos_vals[:n]
        vel_vals = vel_vals[:n]

        csv_max_pos = float(np.max(pos_vals)) if n > 0 else float(env_S)
        if csv_max_pos <= 0.0:
            csv_max_pos = float(env_S)
        csv_max_steps = float(n)

        dt = 0.2
        acc_vals = np.zeros_like(vel_vals)
        if n > 1:
            acc_vals[1:] = (vel_vals[1:] - vel_vals[:-1]) / dt
        acc_vals = np.clip(acc_vals, float(action_min), float(action_max))

        step_idx = 0
        prev_acc = float(acc_vals[0]) if n > 0 else 0.0

        # 提取限速列
        sl_col = 'EBI速度(km/h)' if 'EBI速度(km/h)' in df.columns else (
            '目标速度（km/h）' if '目标速度（km/h）' in df.columns else None)

        for idx in range(n):
            pos = float(pos_vals[idx])
            veo = float(vel_vals[idx])
            real_acc = float(acc_vals[idx])

            action_idx = int(round((real_acc - float(action_min)) / float(action_step)))
            max_idx = int(round((float(action_max) - float(action_min)) / float(action_step)))
            action_idx = np.clip(action_idx, 0, max_idx)

            s_ratio = pos / csv_max_pos
            v_ratio = veo / env_max_speed if env_max_speed > 0 else 0
            t_ratio = np.clip(float(step_idx) / csv_max_steps, 0.0, 1.0)
            prev_acc_ratio = prev_acc / float(action_max)

            # 【修改点 6】：将真实的限速数据写入状态
            limit_kph = float(df.iloc[idx].get(sl_col, 80.0)) if sl_col is not None else 80.0
            limit_ratio = (limit_kph / 3.6) / env_max_speed

            grad_val = float(df.iloc[idx].get('当前坡度', 0.0))
            grad_ratio = np.clip(grad_val / float(env_max_grad), -1.0, 1.0) if env_max_grad > 0 else 0.0

            state_tuple = (s_ratio, v_ratio, t_ratio, prev_acc_ratio, limit_ratio, grad_ratio)
            action_tuple = (action_idx,)

            expert_states.append(list(state_tuple))
            expert_actions.append(list(action_tuple))

            prev_acc = real_acc
            step_idx += 1



    return np.array(expert_states, dtype=np.float32), np.array(expert_actions, dtype=np.int64)


def load_offline_data(data_dir, env, batch_size, device, action_step=0.2, action_min=-0.8, action_max=0.8):
    states, actions = parse_csv_logs(data_dir, env.S, env.max_speed, env.T, env.max_grad, action_step, action_min, action_max)


    dataset_size = len(states)
    if dataset_size == 0: raise ValueError("未能解析到数据！")

    indices = np.random.permutation(dataset_size)
    states = np.take(states, indices, axis=0)
    actions = np.take(actions, indices, axis=0)
    split = int(dataset_size * 0.8)

    train_states = np.take(states, range(0, split), axis=0)
    train_actions = np.take(actions, range(0, split), axis=0)
    cal_states = np.take(states, range(split, dataset_size), axis=0)
    cal_actions = np.take(actions, range(split, dataset_size), axis=0)

    # 修改成 6 维
    bc_buffer = ReplayBuffer(max_size=max(len(train_states), 1), state_dim=env.n_features, action_dim=1, device=device)
    cal_buffer = ReplayBuffer(max_size=max(len(cal_states), 1), state_dim=env.n_features, action_dim=1, device=device)


    for s, a in zip(train_states, train_actions): bc_buffer.add(s, a, 0.0, s, 0.0)
    for s, a in zip(cal_states, cal_actions): cal_buffer.add(s, a, 0.0, s, 0.0)

    return bc_buffer, cal_buffer