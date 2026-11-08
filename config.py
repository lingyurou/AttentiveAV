"""
AttentiveAV 全局配置
====================
所有超参集中在此，训练/评估脚本统一从这里读，避免散落魔法数字。
"""
from pathlib import Path


class Config:
    def __init__(self):
        # ---------- 路径 ----------
        self.base_dir = Path(__file__).parent.absolute()
        self.lrs2_root = self.base_dir / "data" / "lrs3" / "LRS2数据集"
        self.cache_dir = self.base_dir / "data" / "cache" / "lrs2"
        self.run_dir = self.base_dir / "runs"
        self.result_dir = self.base_dir / "results"
        for d in (self.run_dir, self.result_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ---------- 数据 ----------
        self.sample_rate = 16000
        self.clip_seconds = 2.0
        self.clip_samples = int(self.sample_rate * self.clip_seconds)   # 32000
        self.fps = 25
        self.clip_frames = int(self.fps * self.clip_seconds)            # 50
        self.mouth_size = 96
        # 混合信噪比范围（dB）：越负干扰越强
        self.snr_min = -5.0
        self.snr_max = 5.0
        # 难样本对：基频接近的说话人（视觉线索才真正起作用）
        self.hard_pair_ratio = 0.5
        self.f0_tol = 30.0            # Hz

        # ---------- 模型 ----------
        self.n_filters = 192          # 编码器通道数 N
        self.enc_kernel = 16
        self.enc_stride = 8
        self.tcn_channels = 192       # 分离网络通道 C
        self.tcn_repeats = 3
        self.tcn_dilations = (1, 2, 4, 8)
        self.visual_dim = 192
        self.mask_scale = 1.0
        self.mask_act = "sigmoid"     # relu | sigmoid（频谱损失下 sigmoid 更稳）
        self.residual_mask = False    # True: est = mix - decoder(enc*mask)
        self.loss_lam = 1.0
        self.aux_weight = 0.1         # 视觉->梅尔 蒸馏损失权重
        self.n_mels = 80
        self.mel_hop = 640            # 16k/640 = 25 fps，与视频帧对齐           # 频谱幅度 L1 的权重（与 -SI-SNR 联合）
        self.n_fft = 512
        self.hop = 128

        # ---------- 训练 ----------
        self.batch_size = 8
        self.num_epochs = 30
        self.learning_rate = 1e-3
        self.min_lr = 1e-5
        self.weight_decay = 0.0
        self.grad_clip = 5.0
        self.amp = True
        self.num_workers = 6
        self.seed = 2024

        # ---------- 评估 ----------
        self.eval_every = 1
        self.eval_limit = 0           # 0 = 全量

    def __str__(self):
        return (f"Config(clip={self.clip_seconds}s@{self.sample_rate}Hz, "
                f"N={self.n_filters}, C={self.tcn_channels}, "
                f"dilations={self.tcn_dilations}x{self.tcn_repeats}, "
                f"batch={self.batch_size}, lr={self.learning_rate})")


config = Config()
