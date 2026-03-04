parameters = {
    "seed": 42,
    "state_dim": 6,           # 【史诗级修复 1】：升级为 6 维状态！加入限速+坡度感知！

    "action_min": -0.8,
    "action_max": 0.8,
    "action_step": 0.04,
    "num_actions": 41,



    "max_speed": 80 / 3.6,

    "max_episodes": 2000,
    "device": "cuda",
    "discount": 0.99,
    "lr": 0.0005,
    "tau": 0.005,
    "epsilon_start": 0.6,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.995,
    "online_buffer_size": 20000,
    "batch_size": 256,
    "confidence_level": 0.90, # 提高置信度以降低阈值
    "phase_bins": (0.15, 0.85),  # 起步/巡航/进站 分段阈值（按 s_ratio 切分）
    "phase_min_samples": 200,    # 每段校准最少样本，太少则回退全局阈值
    "phase_confidence": (0.95, 0.99, 0.99),  # 分阶段置信度（巡航/进站更高→阈值更低）

    "temperature": 1.0,          # 概率分布温度（越大越平缓）

    "entropy_weight": 0.01,        # 分阶段熵正则强度（提高以平缓分布）
    "entropy_targets": (2.5, 3.6, 3.2),  # 起步/巡航/进站 目标熵（ln(41)≈3.71）

    "prior_weight": 0.01,       # 分阶段动作先验约束（降低以减少尖峰）

    "bc_weight": 20.0,

    "l2_reg": 0.01,
    "layer_size": 256
}