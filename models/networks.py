import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQNSF(nn.Module):
    def __init__(self, state_dim, num_actions, layer_size=256, temperature=2.0):
        super(DDQNSF, self).__init__()
        self.q1 = nn.Linear(state_dim, layer_size)
        self.q2 = nn.Linear(layer_size, layer_size)
        self.q3 = nn.Linear(layer_size, num_actions)

        self.i1 = nn.Linear(state_dim, layer_size)
        self.i2 = nn.Linear(layer_size, layer_size)
        self.i3 = nn.Linear(layer_size, num_actions)

        # 【神级修复 4】：温度系数，让输出的概率分布变平缓
        self.temperature = float(temperature)


    def forward(self, state):
        q = F.relu(self.q1(state))
        q = F.relu(self.q2(q))
        q_vals = self.q3(q)

        i = F.relu(self.i1(state))
        i = F.relu(self.i2(i))
        logits = self.i3(i)

        # 使用温度缩放
        soft_logits = logits / self.temperature

        return q_vals, F.log_softmax(soft_logits, dim=1), soft_logits