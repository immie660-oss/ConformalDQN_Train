# utils/buffer.py
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim=1, device="cuda"):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device

        # 核心加速：建在内存中(numpy)，极速读写，告别显卡通讯延迟
        self.state = np.zeros(tuple((self.max_size, state_dim)), dtype=np.float32)
        self.action = np.zeros(tuple((self.max_size, action_dim)), dtype=np.int64)
        self.reward = np.zeros(tuple((self.max_size, 1)), dtype=np.float32)
        self.next_state = np.zeros(tuple((self.max_size, state_dim)), dtype=np.float32)
        self.not_done = np.zeros(tuple((self.max_size, 1)), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        idx = self.ptr

        self.state.__setitem__(idx, state)
        self.next_state.__setitem__(idx, next_state)

        try:
            len(action)
            self.action.__setitem__(idx, action)
        except TypeError:
            self.action.__setitem__(idx, tuple((int(action),)))

        try:
            len(reward)
            self.reward.__setitem__(idx, reward)
        except TypeError:
            self.reward.__setitem__(idx, tuple((float(reward),)))

        try:
            len(done)
            d_val = float(done.__getitem__(0))
        except TypeError:
            d_val = float(done)
        self.not_done.__setitem__(idx, tuple((1.0 - d_val,)))

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # 极速切片
        ind = np.random.randint(0, self.size, size=batch_size)

        s = np.take(self.state, ind, axis=0)
        a = np.take(self.action, ind, axis=0)
        r = np.take(self.reward, ind, axis=0)
        nd = np.take(self.not_done, ind, axis=0)
        ns = np.take(self.next_state, ind, axis=0)

        # 只有在要训练时，才“满载一卡车”发给显卡
        return (
            torch.FloatTensor(s).to(self.device),
            torch.LongTensor(a).to(self.device),
            torch.FloatTensor(r).to(self.device),
            torch.FloatTensor(nd).to(self.device),
            torch.FloatTensor(ns).to(self.device)
        )

    def get_all(self):
        ind = list(range(self.size))
        s = np.take(self.state, ind, axis=0)
        a = np.take(self.action, ind, axis=0)
        return (
            torch.FloatTensor(s).to(self.device),
            torch.LongTensor(a).to(self.device)
        )