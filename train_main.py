import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.nn.functional as F


torch.backends.cudnn.benchmark = True

from parameters import parameters
from env.subway_env import TrainLine
from env import TrainAndRoadCharacter as trc
from utils.data_processor import load_offline_data
from utils.buffer import ReplayBuffer
from models.conformal_dqn import ConformalDQN
from models.conformal_predictor import ConformalPredictor
from utils.track_parser import TrackProfile


def plot_full_trajectory(pos_list, vel_list, sl_x, sl_y, out_dir, file_name,
                         grad_x=None, grad_y=None, grad_scale=0.5):

    fig, ax1 = plt.subplots(figsize=tuple((15, 6)))
    ax1.plot(pos_list, vel_list, color='blue', linewidth=1.5, label='DQN Velocity')


    sl_x_arr = np.asarray(sl_x, dtype=float)
    sl_y_arr = np.asarray(sl_y, dtype=float)
    mask = np.isfinite(sl_x_arr) & np.isfinite(sl_y_arr)
    if mask.any():
        sl_x_arr = sl_x_arr[mask]
        sl_y_arr = sl_y_arr[mask]
        ax1.plot(sl_x_arr, sl_y_arr, color='red', linestyle='--', linewidth=2, label='EBI Speed Limit')

    ax2 = None
    if grad_x is not None and grad_y is not None:
        grad_x_arr = np.asarray(grad_x, dtype=float)
        grad_y_arr = np.asarray(grad_y, dtype=float)
        gmask = np.isfinite(grad_x_arr) & np.isfinite(grad_y_arr)
        if gmask.any():
            grad_x_arr = grad_x_arr[gmask]
            grad_y_arr = grad_y_arr[gmask] * float(grad_scale)
            ax2 = ax1.twinx()
            ax2.plot(grad_x_arr, grad_y_arr, color='#FF8C00', linewidth=1.2, alpha=0.85, label='Gradient')
            ax2.set_ylabel("Gradient (‰) x" + str(grad_scale))

    ax1.set_title("Full Line (13-25) Velocity - Distance (" + file_name + ")")
    ax1.set_xlabel("Total Distance (m)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_ylim(bottom=0.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend()
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, file_name + "_v_s.png"))
    plt.close(fig)


def plot_expert_trajectory(pos_list, vel_list, sl_x, sl_y, out_dir, file_name,
                           grad_x=None, grad_y=None, grad_scale=0.5):
    if pos_list is None or vel_list is None or len(pos_list) == 0:
        return
    fig, ax1 = plt.subplots(figsize=tuple((15, 6)))
    ax1.plot(pos_list, vel_list, color='green', linewidth=1.5, label='Expert Velocity')

    sl_x_arr = np.asarray(sl_x, dtype=float)
    sl_y_arr = np.asarray(sl_y, dtype=float)
    mask = np.isfinite(sl_x_arr) & np.isfinite(sl_y_arr)
    if mask.any():
        sl_x_arr = sl_x_arr[mask]
        sl_y_arr = sl_y_arr[mask]
        ax1.plot(sl_x_arr, sl_y_arr, color='red', linestyle='--', linewidth=2, label='EBI Speed Limit')

    ax2 = None
    if grad_x is not None and grad_y is not None:
        grad_x_arr = np.asarray(grad_x, dtype=float)
        grad_y_arr = np.asarray(grad_y, dtype=float)
        gmask = np.isfinite(grad_x_arr) & np.isfinite(grad_y_arr)
        if gmask.any():
            grad_x_arr = grad_x_arr[gmask]
            grad_y_arr = grad_y_arr[gmask] * float(grad_scale)
            ax2 = ax1.twinx()
            ax2.plot(grad_x_arr, grad_y_arr, color='#FF8C00', linewidth=1.2, alpha=0.85, label='Gradient')
            ax2.set_ylabel("Gradient (‰) x" + str(grad_scale))

    ax1.set_title("Expert Velocity - Distance (" + file_name + ")")
    ax1.set_xlabel("Total Distance (m)")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.set_ylim(bottom=0.0)

    handles1, labels1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend()
    ax1.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, file_name + "_v_s.png"))
    plt.close(fig)




def plot_action_probs_filter(action_probs, threshold, out_dir, file_name, phase_name, action_step, action_min):

    num_actions = int(len(action_probs))
    actions = np.arange(num_actions)
    accels = np.round((actions * float(action_step)) + float(action_min), 2)


    safe_set = list()
    for i in range(num_actions):
        if action_probs.item(i) >= threshold:
            safe_set.append(i)


    if len(safe_set) == 0:
        max_idx = int(np.argmax(action_probs))
        safe_set.append(max_idx)

    plt.figure(figsize=tuple((8, 5)))
    colors = list()
    for i in range(num_actions):
        if i in safe_set:
            colors.append('green')
        else:
            colors.append('red')


    plt.bar(actions, action_probs, color=colors, alpha=0.8)

    labels = list()
    for i in range(num_actions):
        labels.append(str(i) + "\n(" + str(accels.item(i)) + ")")

    if num_actions > 15:
        step_tick = max(1, num_actions // 8)
        tick_idx = actions[::step_tick]
        plt.xticks(tick_idx, tuple(labels[::step_tick]))
    else:
        plt.xticks(actions, tuple(labels))

    plt.xlabel("Action Index (Acceleration m/s^2)")
    plt.ylabel("Predicted Probability")
    plt.title("Conditional Threshold CP - " + phase_name)

    safe_patch = mpatches.Patch(color='green', label='Safe Set (Prob >= ' + str(round(threshold, 2)) + ')')
    unsafe_patch = mpatches.Patch(color='red', label='Unsafe (Filtered)')
    plt.legend(handles=list((safe_patch, unsafe_patch)))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, file_name + ".png"))
    plt.close()


def load_offline_accs(data_dir, action_min, action_max):
    s_ratios = list()
    accs = list()


    for file in os.listdir(data_dir):
        if not file.endswith('.csv'):
            continue
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
        if pos_vals.size == 0 or vel_vals.size == 0:
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

        max_pos = float(np.max(pos_vals)) if n > 0 else 1.0
        if max_pos <= 0:
            max_pos = 1.0

        dt = 0.2
        real_acc = np.zeros_like(vel_vals)
        if n > 1:
            real_acc[1:] = (vel_vals[1:] - vel_vals[:-1]) / dt
        real_acc = np.clip(real_acc, float(action_min), float(action_max))

        s_ratio = pos_vals / max_pos

        s_ratios.append(s_ratio)
        accs.append(real_acc)


    if len(s_ratios) == 0:
        return np.array([]), np.array([])

    return np.concatenate(s_ratios), np.concatenate(accs)


def load_expert_full_line(data_dir, start_idx=13, end_idx=25):
    full_pos = list()
    full_vel = list()
    global_offset = 0.0

    for i in range(int(start_idx), int(end_idx) + 1):
        csv_path = os.path.join(data_dir, str(i) + ".csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig', encoding_errors='ignore', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path, encoding='gbk', encoding_errors='ignore', on_bad_lines='skip')

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
        if vel_vals.size == 0:
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

        if pos_vals.size == 0:
            continue

        for p_val, v_val in zip(pos_vals, vel_vals):
            full_pos.append(float(p_val) + global_offset)
            full_vel.append(float(v_val))

        max_pos = float(np.max(pos_vals))
        if max_pos <= 0.0:
            max_pos = float(pos_vals[-1]) if pos_vals.size > 0 else 0.0
        global_offset += max_pos

    return full_pos, full_vel


def load_ebi_limit_full_line(data_dir, start_idx=13, end_idx=25):
    full_pos = list()
    full_lim = list()
    global_offset = 0.0

    for i in range(int(start_idx), int(end_idx) + 1):
        csv_path = os.path.join(data_dir, str(i) + ".csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig', encoding_errors='ignore', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path, encoding='gbk', encoding_errors='ignore', on_bad_lines='skip')

        if '计算距离(m)' not in df.columns:
            continue

        sl_col = None
        for cand in ('EBI速度(km/h)', 'EBI速度', '目标速度（km/h）', '目标速度(km/h)', '目标速度'):
            if cand in df.columns:
                sl_col = cand
                break
        if sl_col is None:
            continue

        pos_vals = df['计算距离(m)'].to_numpy(dtype=float)
        lim_vals = df[sl_col].to_numpy(dtype=float)
        if pos_vals.size == 0 or lim_vals.size == 0:
            continue

        lim_max = float(np.nanmax(lim_vals)) if np.isfinite(lim_vals).any() else 0.0
        if 'km/h' in sl_col:
            lim_vals = lim_vals / 3.6
        elif lim_max > 60.0:
            lim_vals = lim_vals / 3.6

        n = int(min(len(pos_vals), len(lim_vals)))
        pos_vals = pos_vals[:n]
        lim_vals = lim_vals[:n]

        max_pos = float(np.nanmax(pos_vals)) if np.isfinite(pos_vals).any() else 0.0
        if max_pos <= 0.0:
            max_pos = float(pos_vals[-1]) if pos_vals.size > 0 else 0.0

        full_pos.append(global_offset)
        full_lim.append(0.0)

        for p_val, l_val in zip(pos_vals, lim_vals):
            if not (np.isfinite(p_val) and np.isfinite(l_val)):
                continue
            full_pos.append(float(p_val) + global_offset)
            full_lim.append(float(l_val))

        full_pos.append(global_offset + max_pos)
        full_lim.append(0.0)

        global_offset += max_pos


    return full_pos, full_lim


def load_grad_full_line(data_dir, start_idx=13, end_idx=25):
    full_pos = list()
    full_grad = list()
    global_offset = 0.0

    for i in range(int(start_idx), int(end_idx) + 1):
        csv_path = os.path.join(data_dir, str(i) + ".csv")
        if not os.path.exists(csv_path):
            continue

        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig', encoding_errors='ignore', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path, encoding='gbk', encoding_errors='ignore', on_bad_lines='skip')

        if '计算距离(m)' not in df.columns or '当前坡度' not in df.columns:
            continue

        pos_vals = df['计算距离(m)'].to_numpy(dtype=float)
        grad_vals = df['当前坡度'].to_numpy(dtype=float)
        if pos_vals.size == 0 or grad_vals.size == 0:
            continue

        n = int(min(len(pos_vals), len(grad_vals)))
        pos_vals = pos_vals[:n]
        grad_vals = grad_vals[:n]

        max_pos = float(np.nanmax(pos_vals)) if np.isfinite(pos_vals).any() else 0.0
        if max_pos <= 0.0:
            max_pos = float(pos_vals[-1]) if pos_vals.size > 0 else 0.0

        for p_val, g_val in zip(pos_vals, grad_vals):
            if not (np.isfinite(p_val) and np.isfinite(g_val)):
                continue
            full_pos.append(float(p_val) + global_offset)
            full_grad.append(float(g_val))

        full_pos.append(float('nan'))
        full_grad.append(float('nan'))
        global_offset += max_pos

    return full_pos, full_grad




def plot_phase_action_dist_005(s_ratios, accs, phase_bins, out_dir, action_min, action_max, action_step):

    if len(s_ratios) == 0:
        print("离线动作分布统计失败：未加载到有效数据。")
        return

    min_a, max_a, step = float(action_min), float(action_max), float(action_step)
    edges = np.arange(min_a, max_a + step * 0.5, step)
    centers = edges[:-1] + step / 2.0


    phase_names = ("Acceleration", "Cruising", "Braking")
    ranges = list(((0.0, float(phase_bins[0])), (float(phase_bins[0]), float(phase_bins[1])), (float(phase_bins[1]), 1.1)))

    for pid, (lo, hi) in enumerate(ranges):
        mask = (s_ratios >= lo) & (s_ratios < hi)
        if not np.any(mask):
            print("离线动作分布统计：" + phase_names[pid] + " 样本为 0。")
            continue

        acc_phase = accs[mask]
        counts, _ = np.histogram(acc_phase, bins=edges)
        total = counts.sum()
        if total == 0:
            continue
        probs = counts / total
        cdf = np.cumsum(probs)

        plt.figure(figsize=tuple((10, 4.5)))
        plt.bar(centers, probs, width=step * 0.9, color='#4C78A8', alpha=0.85)
        plt.xlabel("Acceleration bin (m/s^2)")
        plt.ylabel("Probability")
        step_label = f"{float(step):.2f}"
        step_tag = step_label.replace(".", "p")
        plt.title("Offline Action Distribution (" + step_label + ") - " + phase_names[pid])
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ActionDist_" + step_tag + "_" + phase_names[pid] + "_Hist.png"))
        plt.close()

        plt.figure(figsize=tuple((10, 4.5)))
        plt.plot(centers, cdf, color='#F58518', linewidth=2)
        plt.ylim((0.0, 1.05))
        plt.xlabel("Acceleration bin (m/s^2)")
        plt.ylabel("CDF")
        plt.title("Offline Action CDF (" + step_label + ") - " + phase_names[pid])
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ActionDist_" + step_tag + "_" + phase_names[pid] + "_CDF.png"))
        plt.close()




def print_offline_phase_top1(agent, offline_bc_buffer, phase_bins, label="离线"):
    states, actions = offline_bc_buffer.get_all()
    s_ratio = states[:, 0]

    with torch.no_grad():
        _, log_probs, _ = agent.Q(states)
        pred_top1 = log_probs.exp().argmax(dim=1)

    phase_names = ("起步", "巡航", "进站")
    ranges = list(((0.0, float(phase_bins[0])), (float(phase_bins[0]), float(phase_bins[1])), (float(phase_bins[1]), 1.1)))

    for pid, (lo, hi) in enumerate(ranges):
        mask = (s_ratio >= lo) & (s_ratio < hi)
        count = int(mask.sum().item())
        if count == 0:
            print(label + phase_names[pid] + "段样本为 0，无法统计 top1。")
            continue

        idx = mask.nonzero(as_tuple=False).squeeze(1)
        pred_phase = pred_top1.index_select(0, idx).cpu()
        act_phase = actions.squeeze(1).index_select(0, idx).cpu()

        pred_counts = torch.bincount(pred_phase, minlength=agent.num_actions).float()
        act_counts = torch.bincount(act_phase, minlength=agent.num_actions).float()

        pred_top = int(pred_counts.argmax().item())
        act_top = int(act_counts.argmax().item())

        pred_ratio = float((pred_counts / pred_counts.sum()).max().item())
        act_ratio = float((act_counts / act_counts.sum()).max().item())

        print(
            label + phase_names[pid] + "段 top1 统计 | 预测 top1: "
            + str(pred_top)
            + " (占比 "
            + str(round(pred_ratio, 3))
            + ") | 专家 top1: "
            + str(act_top)
            + " (占比 "
            + str(round(act_ratio, 3))
            + ")"
        )


def plot_cp_from_offline(agent, cp, offline_bc_buffer, phase_bins, out_dir, episode_tag="", action_step=0.2, action_min=-0.8):

    states, _ = offline_bc_buffer.get_all()
    states_cpu = states.detach().cpu().numpy()

    phase_names = ("Acceleration", "Cruising", "Braking")
    ranges = list(((0.0, float(phase_bins[0])), (float(phase_bins[0]), float(phase_bins[1])), (float(phase_bins[1]), 1.1)))

    for pid, (lo, hi) in enumerate(ranges):
        idxs = np.where((states_cpu[:, 0] >= lo) & (states_cpu[:, 0] < hi))[0]
        if len(idxs) == 0:
            continue

        s_vals = states_cpu[idxs, 0]
        mid_idx = idxs[np.argsort(s_vals)[len(idxs) // 2]]

        state_tensor = states[mid_idx].unsqueeze(0)
        state_np = states_cpu[mid_idx]

        with torch.no_grad():
            _, action_log_probs, _ = agent.Q(state_tensor)
            probs = action_log_probs.exp().squeeze(0).cpu().numpy()

        threshold = cp.get_threshold(state_np)
        tag = str(episode_tag).strip()
        if tag == "":
            file_name = "CP_Offline_" + phase_names[pid]
        elif tag.lower() == "pretrain":
            file_name = "CP_Offline_Pretrain_" + phase_names[pid]
        else:
            file_name = "CP_Offline_Ep" + tag + "_" + phase_names[pid]
        plot_action_probs_filter(
            probs,
            threshold,
            out_dir,
            file_name,
            phase_names[pid] + "(Offline)",
            action_step,
            action_min
        )



        top_idx = int(np.argmax(probs))
        top_prob = float(np.max(probs))
        print(
            "离线CP采样|" + phase_names[pid]
            + " s=" + str(round(float(state_np[0]), 3))
            + " v=" + str(round(float(state_np[1]), 3))
            + " t=" + str(round(float(state_np[2]), 3))
            + " u=" + str(round(float(state_np[3]), 3))
            + " lim=" + str(round(float(state_np[4]), 3))
            + " grad=" + str(round(float(state_np[5]), 3))
            + " | top=" + str(top_idx)
            + " p=" + str(round(top_prob, 3))
            + " | p*=" + str(round(float(threshold), 3))
        )










def main():
    dev = parameters.get("device")
    batch_sz = parameters.get("batch_size")
    action_min = float(parameters.get("action_min", -0.8))
    action_max = float(parameters.get("action_max", 0.8))
    action_step = float(parameters.get("action_step", 0.2))
    num_actions = int(parameters.get("num_actions", int(round((action_max - action_min) / action_step)) + 1))

    img_dir = "./img"
    results_dir = "./results"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    track_profiles = list()
    for i in range(13, 26):
        csv_path = "./data/" + str(i) + ".csv"
        if os.path.exists(csv_path): track_profiles.append(TrackProfile(csv_path))

    track_profiles.__getitem__(0).apply_to_env()
    env = TrainLine(
        time=track_profiles.__getitem__(0).T,
        action_step=action_step,
        action_min=action_min,
        action_max=action_max,
        num_actions=num_actions,
        time_limit=track_profiles.__getitem__(0).T
    )



    offline_bc_buffer, cal_buffer = load_offline_data(
        "./data",
        env,
        batch_sz,
        dev,
        action_step,
        action_min,
        action_max
    )
    s_ratios, accs = load_offline_accs("./data", action_min, action_max)

    plot_phase_action_dist_005(s_ratios, accs, parameters.get("phase_bins"), img_dir, action_min, action_max, action_step)

    expert_pos, expert_vel = load_expert_full_line("./data", 13, 25)
    ebi_pos, ebi_lim = load_ebi_limit_full_line("./data", 13, 25)
    grad_pos, grad_vals = load_grad_full_line("./data", 13, 25)

    plot_expert_trajectory(
        expert_pos,
        expert_vel,
        ebi_pos,
        ebi_lim,
        img_dir,
        "Expert_Velocity_FullLine",
        grad_x=grad_pos,
        grad_y=grad_vals
    )



    online_buffer = ReplayBuffer(max_size=parameters.get("online_buffer_size"), state_dim=parameters.get("state_dim"), action_dim=1, device=dev)





    agent = ConformalDQN(parameters)
    cp = ConformalPredictor(
        agent,
        cal_buffer,
        confidence_level=parameters.get("confidence_level"),
        phase_bins=parameters.get("phase_bins"),
        min_samples=parameters.get("phase_min_samples"),
        phase_confidence=parameters.get("phase_confidence")
    )




    agent.conformal_predictor = cp

    print("预训练中...")
    for _ in range(5000): agent.train(offline_bc_buffer, offline_bc_buffer, batch_sz)
    cp.calibrate()
    print_offline_phase_top1(agent, offline_bc_buffer, parameters.get("phase_bins"), label="离线预训练")
    plot_cp_from_offline(
        agent,
        cp,
        offline_bc_buffer,
        parameters.get("phase_bins"),
        img_dir,
        episode_tag="pretrain",
        action_step=action_step,
        action_min=action_min
    )




    best_eval_reward = -float('inf')


    for episode in range(parameters.get("max_episodes")):
        rand_idx = np.random.randint(0, len(track_profiles))
        tp = track_profiles.__getitem__(rand_idx)
        tp.apply_to_env()
        env = TrainLine(
            time=tp.T,
            action_step=action_step,
            action_min=action_min,
            action_max=action_max,
            num_actions=num_actions,
            time_limit=tp.T
        )


        state = env.reset()


        step_count = 0
        episode_reward = 0.0
        while True:
            step_count += 1
            # 【提取6维状态】
            norm_state = list((state.item(0), state.item(1), state.item(2), state.item(3) / action_max, state.item(4), state.item(5)))

            action = agent.select_action(norm_state, eval_mode=False)

            next_state, ec, reward, done, real_act = env.step(action)
            norm_next_state = list((
                                   next_state.item(0), next_state.item(1), next_state.item(2), next_state.item(3) / action_max,
                                   next_state.item(4), next_state.item(5)))



            online_buffer.add(norm_state, real_act, reward, norm_next_state, done)
            state = next_state
            episode_reward += reward
            if done: break

        if online_buffer.size > batch_sz:
            for _ in range(step_count // 4): agent.train(online_buffer, offline_bc_buffer, batch_sz)

        if episode % 10 == 0:
            print("Episode: " + str(episode) + " Train Reward: " + str(round(episode_reward, 2)))

        if episode > 0 and episode % 50 == 0:
            full_pos, full_vel = list(), list()
            full_sl_x, full_sl_y = list(), list()
            global_offset = 0.0
            eval_reward = 0.0
            phase_stats = {
                0: {"steps": 0, "safe": 0, "maxp": 0.0},
                1: {"steps": 0, "safe": 0, "maxp": 0.0},
                2: {"steps": 0, "safe": 0, "maxp": 0.0}
            }

            for t_idx in range(len(track_profiles)):

                tp = track_profiles.__getitem__(t_idx)
                tp.apply_to_env()
                env = TrainLine(
                    time=tp.T,
                    action_step=action_step,
                    action_min=action_min,
                    action_max=action_max,
                    num_actions=num_actions,
                    time_limit=tp.T
                )


                eval_state = env.reset()

                eval_step = 0
                while True:



                    eval_step += 1
                    norm_eval_state = list((eval_state.item(0), eval_state.item(1), eval_state.item(2),
                                            eval_state.item(3) / action_max, eval_state.item(4), eval_state.item(5)))



                    state_tensor = torch.FloatTensor(norm_eval_state).unsqueeze(0).to(dev)
                    with torch.no_grad():
                        _, action_log_probs, _ = agent.Q(state_tensor)
                        probs = action_log_probs.exp().squeeze(0).cpu().numpy()

                    threshold = cp.get_threshold(norm_eval_state)
                    safe_mask = probs >= threshold
                    s_ratio = float(norm_eval_state[0])
                    if s_ratio < cp.phase_bins[0]:
                        phase_id = 0
                    elif s_ratio < cp.phase_bins[1]:
                        phase_id = 1
                    else:
                        phase_id = 2

                    safe_count = int(safe_mask.sum())
                    max_prob = float(probs.max())

                    phase_stats[phase_id]["steps"] += 1
                    phase_stats[phase_id]["safe"] += safe_count
                    phase_stats[phase_id]["maxp"] += max_prob

                    action = agent.select_action(norm_eval_state, eval_mode=True)


                    next_eval_state, _, r, d, _ = env.step(action)
                    eval_reward += r

                    curr_real_pos = float(next_eval_state.item(0)) * tp.S + global_offset
                    full_pos.append(curr_real_pos)
                    full_vel.append(float(next_eval_state.item(1)) * env.max_speed)

                    eval_state = next_eval_state



                    if d:
                        if t_idx == len(track_profiles) - 1:
                            print("      - 最后区间结束原因: " + str(getattr(env, "done_reason", "unknown"))
                                  + " | pos=" + str(round(float(env.pos), 2))
                                  + " / S=" + str(round(float(tp.S), 2))
                                  + " | t=" + str(round(float(getattr(env, "t_current", 0.0)), 2)))
                        full_pos.append(float('nan'))
                        full_vel.append(float('nan'))
                        break


                global_offset += tp.S



            print("  >>> Episode " + str(episode) + " 全线联调得分: " + str(round(eval_reward, 2)))

            for pid, name in ((0, "起步"), (1, "巡航"), (2, "进站")):
                steps = phase_stats[pid]["steps"]
                if steps > 0:
                    avg_safe = phase_stats[pid]["safe"] / steps
                    avg_maxp = phase_stats[pid]["maxp"] / steps
                    print("      - " + name + " 平均安全动作数: " + str(round(avg_safe, 2))
                          + " | 平均最大概率: " + str(round(avg_maxp, 3)))

            plot_full_trajectory(
                full_pos,
                full_vel,
                ebi_pos,
                ebi_lim,
                img_dir,
                "Full_Line_Ep" + str(episode),
                grad_x=grad_pos,
                grad_y=grad_vals
            )


            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                plot_full_trajectory(
                    full_pos,
                    full_vel,
                    ebi_pos,
                    ebi_lim,
                    img_dir,
                    "Full_Line_BestReward_Ep" + str(episode),
                    grad_x=grad_pos,
                    grad_y=grad_vals
                )





            print_offline_phase_top1(agent, offline_bc_buffer, parameters.get("phase_bins"), label="离线@Ep" + str(episode) + "-")
            plot_cp_from_offline(
                agent,
                cp,
                offline_bc_buffer,
                parameters.get("phase_bins"),
                img_dir,
                episode_tag=str(episode),
                action_step=action_step,
                action_min=action_min
            )







if __name__ == "__main__":
    main()