import pandas as pd
import numpy as np
from env import TrainAndRoadCharacter as trc


class TrackProfile:
    def __init__(self, csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='gbk', on_bad_lines='skip')
        except Exception:
            df = pd.read_csv(csv_path, encoding='utf-8-sig', on_bad_lines='skip')

        # 解析真实人类时间
        time_series = df.get('时间').tolist()

        def parse_time(t_str):
            parts = str(t_str).split(':')
            return int(parts.__getitem__(0)) * 3600 + int(parts.__getitem__(1)) * 60 + int(parts.__getitem__(2))

        start_t = parse_time(time_series.__getitem__(0))
        end_t = parse_time(time_series.__getitem__(len(time_series) - 1))
        self.T = float(end_t - start_t)
        if self.T <= 0.0:
            self.T = 110.0  # 兜底

        # 解析真实距离
        self.S = float(df.get('计算距离(m)').max())

        # 解析限速曲线
        sl_col = 'EBI速度(km/h)' if 'EBI速度(km/h)' in df.keys() else '目标速度（km/h）'
        self.sl_pts = list()
        self.sl_spds = list()

        prev_sl = -1.0
        for i in range(len(df)):
            curr_sl = float(df.get(sl_col).__getitem__(i))
            if curr_sl != prev_sl:
                self.sl_pts.append(float(df.get('计算距离(m)').__getitem__(i)))
                self.sl_spds.append(curr_sl)
                prev_sl = curr_sl

        # 解析坡度曲线
        self.grad_pts = list()
        self.grad_vals = list()
        prev_grad = -999.0
        for i in range(len(df)):
            curr_grad = float(df.get('当前坡度').__getitem__(i))
            if curr_grad != prev_grad:
                self.grad_pts.append(float(df.get('计算距离(m)').__getitem__(i)))
                self.grad_vals.append(curr_grad)
                prev_grad = curr_grad

    def apply_to_env(self):
        # 动态替换物理环境内存中的参数，不修改原文件
        sl_pts = list(self.sl_pts)
        sl_spds = list(self.sl_spds)
        if len(sl_pts) == 0:
            sl_pts = [0.0, self.S]
            sl_spds = [0.0, 0.0]
        else:
            if sl_pts[0] > 0.0:
                sl_pts.insert(0, 0.0)
                sl_spds.insert(0, sl_spds[0])
            if sl_pts[-1] < self.S:
                sl_pts.append(self.S)
                sl_spds.append(sl_spds[-1])

        grad_pts = list(self.grad_pts)
        grad_vals = list(self.grad_vals)
        if len(grad_pts) == 0:
            grad_pts = [0.0, self.S]
            grad_vals = [0.0, 0.0]
        else:
            if grad_pts[0] > 0.0:
                grad_pts.insert(0, 0.0)
                grad_vals.insert(0, grad_vals[0])
            if grad_pts[-1] < self.S:
                grad_pts.append(self.S)
                grad_vals.append(grad_vals[-1])

        trc.SLStartPoint = sl_pts
        trc.speedLimit = sl_spds
        trc.gradStartPoint = grad_pts
        trc.gradList = grad_vals

        # 魔法拦截器：让师兄的代码读取我们动态提取的限速
        def mock_getEBS(pos):
            idx = 0
            for i in range(len(sl_pts)):
                if pos >= sl_pts.__getitem__(i):
                    idx = i
            return sl_spds.__getitem__(idx) / 3.6

        trc.TrainAndRoadData.getEmerencyBrakeSpeed = lambda self_obj, pos: mock_getEBS(pos)
        trc.getRoadspeedLimit = mock_getEBS

        # 同理替换坡度
        def mock_getGrad(pos):
            idx = 0
            for i in range(len(grad_pts)):
                if pos >= grad_pts.__getitem__(i):
                    idx = i
            return grad_vals.__getitem__(idx)

        trc.getRoadGradinet = mock_getGrad
