"""
AttentiveAV 模型族：视觉引导 vs 注册语音引导 的说话人提取
==========================================================
统一范式（Conv-TasNet 风格）：

    波形 --编码器--> 特征 --TCN 分离网络--> 线索调制 --> 掩膜 --解码器--> 目标波形

四种模型共享同一套编码器/TCN/掩膜头/解码器，只替换"线索"与"融合方式"，
因此相互之间的差值就是干净的消融量：

    AttentiveAVModel   (av)          视觉帧特征 + 跨模态注意力 + 梅尔蒸馏
    AudioOnlyModel     (audio)       无线索（退化基线）
    EnrollModel        (enroll)      同说话人参考语音 d-vector + FiLM（VoiceFilter 风格）
    av_noattn                    去跨模态注意力（仅拼接）
    av_nodistill                 去梅尔蒸馏辅助头
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def prep_video(v):
    """uint8 [B,1,F,H,W] -> float [-1,1]（在 GPU 上做，省掉 worker 里的 46 万元素转换）"""
    if v.dtype == torch.uint8:
        v = v.float().div_(255.0).sub_(0.5).mul_(2.0)
    return v


# ---------------------------------------------------------------- 音频主干
class AudioEncoder(nn.Module):
    """一维卷积编码器（Conv-TasNet 风格，可学习基，无激活）"""

    def __init__(self, n_filters=192, kernel=16, stride=8):
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, kernel, stride=stride, bias=False)

    def forward(self, x):
        return self.conv(x)


class AudioDecoder(nn.Module):
    def __init__(self, n_filters=192, kernel=16, stride=8):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(n_filters, 1, kernel, stride=stride, bias=False)

    def forward(self, feat):
        return self.deconv(feat)


class TCNBlock(nn.Module):
    """膨胀深度可分离卷积块（Conv-TasNet 的 1x1 -> PReLU -> D-Conv -> BN -> 残差）"""

    def __init__(self, channels, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, 1)
        self.prelu1 = nn.PReLU()
        self.dconv = nn.Conv1d(channels, channels, kernel, dilation=dilation,
                               padding=pad, groups=channels)
        self.prelu2 = nn.PReLU()
        self.norm = nn.BatchNorm1d(channels)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        y = self.conv1(x)
        y = self.prelu1(y)
        y = self.dconv(y)
        y = self.prelu2(y)
        y = self.norm(y)
        return x + self.res_scale * y


class SeparationNet(nn.Module):
    """TCN 分离网络：把编码特征变成分离特征"""

    def __init__(self, channels=192, repeats=3, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TCNBlock(channels, 3, d) for _ in range(repeats) for d in dilations]
        )
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.out(x)


# ---------------------------------------------------------------- 视觉主干
class VisualEncoder(nn.Module):
    """口型视频 -> 逐帧视觉嵌入（3D 卷积，输出 [B, C, T_v]）"""

    def __init__(self, out_channels=192, mid=64):
        super().__init__()
        self.front = nn.Sequential(
            nn.Conv3d(1, 16, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, mid, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(mid), nn.ReLU(inplace=True),
        )
        self.temporal = nn.Conv1d(mid, out_channels, 5, padding=2)

    def forward(self, video):
        # video: [B, 1, Tv, H, W]  float
        x = self.front(video)                       # [B, mid, Tv, h, w]
        x = x.mean(dim=(3, 4))                      # 口型区域全局池化 -> [B, mid, Tv]
        x = F.relu(self.temporal(x))                # [B, C, Tv]
        return x


class SpeakerEncoder(nn.Module):
    """参考语音 -> d-vector（注册式基线的线索提取器）"""

    def __init__(self, out_dim=192, mid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, mid, 5, stride=2, padding=2), nn.BatchNorm1d(mid), nn.ReLU(inplace=True),
            nn.Conv1d(mid, mid, 5, stride=2, padding=2), nn.BatchNorm1d(mid), nn.ReLU(inplace=True),
            nn.Conv1d(mid, out_dim, 1),
        )

    def forward(self, wav):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        h = self.net(wav)                            # [B, D, T]
        return h.mean(dim=-1)                        # [B, D]


# ---------------------------------------------------------------- 掩膜头
class MaskHead(nn.Module):
    """掩膜预测头。

    relu   : Conv-TasNet 原版，掩膜非负有界但不封顶，起点≈直通，梯度信号强
    sigmoid: 取值 (0,1)，起点接近常数，容易掉进"恒等映射"平台期（实测卡在 0 dB）
    """

    def __init__(self, in_ch, out_ch, act="relu"):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 1)
        self.act = act
        nn.init.zeros_(self.conv.bias)
        if act == "sigmoid":
            self.conv.bias.data.fill_(2.0)

    def forward(self, x):
        y = self.conv(x)
        return torch.relu(y) if self.act == "relu" else torch.sigmoid(y)


# ---------------------------------------------------------------- 完整模型
class AttentiveAVModel(nn.Module):
    """视觉注意力引导的目标说话人提取"""

    def __init__(self, cfg, use_attn=True, use_aux=True):
        super().__init__()
        N, C = cfg.n_filters, cfg.tcn_channels
        self.use_attn = use_attn
        self.use_aux = use_aux
        self.encoder = AudioEncoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.sepnet = SeparationNet(C, cfg.tcn_repeats, cfg.tcn_dilations)
        self.visual = VisualEncoder(C)
        # 跨模态注意力：q 来自视觉，k 来自音频
        self.q_proj = nn.Conv1d(C, C, 1)
        self.k_proj = nn.Conv1d(C, C, 1)
        # 掩膜头同时看音频分离特征与视觉特征（跨模态拼接）
        self.mask_head = MaskHead(2 * C, N, cfg.mask_act)
        # 辅助头：视觉 -> 目标对数梅尔谱（跨模态蒸馏，给视觉分支密集监督）
        self.aux_head = nn.Conv1d(C, 80, 1) if use_aux else None
        self.decoder = AudioDecoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.scale = cfg.mask_scale
        self.stride = cfg.enc_stride
        self.residual = cfg.residual_mask

    def forward(self, mix, video):
        length = mix.shape[-1]
        video = prep_video(video)
        enc = self.encoder(mix)                                 # [B, N, T'] 解码器基
        sep = self.sepnet(enc)                                  # [B, C, T']
        v0 = self.visual(video)                                 # [B, C, Tv]
        v = F.interpolate(v0, size=sep.shape[-1], mode="linear", align_corners=False)

        out = {}
        if self.use_attn:
            q = self.q_proj(v)
            k = self.k_proj(sep)
            attn = torch.sigmoid((q * k).sum(dim=1, keepdim=True) / math.sqrt(sep.shape[1]))
            sep = sep * (1.0 + attn)                            # 视觉逐帧增益
            out["attention_weights"] = attn

        if self.aux_head is not None:
            out["mel_pred"] = self.aux_head(v0)                  # [B, 80, Tv] 辅助蒸馏

        mask = self.scale * self.mask_head(torch.cat([sep, v], dim=1))   # [B, N, T']
        res = self.decoder(enc * mask)                           # 掩膜作用在编码特征上
        # 残差式：est = mix - 干扰估计（本项目默认为 False，保留开关）
        est = mix - res if self.residual else res

        if est.shape[-1] < length:
            est = F.pad(est, (0, length - est.shape[-1]))
        est = est[..., :length]
        out["mask"] = mask
        return est, out


class AudioOnlyModel(nn.Module):
    """同结构、无线索的退化对照组"""

    def __init__(self, cfg):
        super().__init__()
        N, C = cfg.n_filters, cfg.tcn_channels
        self.encoder = AudioEncoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.sepnet = SeparationNet(C, cfg.tcn_repeats, cfg.tcn_dilations)
        self.mask_head = MaskHead(C, N, cfg.mask_act)
        self.decoder = AudioDecoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.scale = cfg.mask_scale
        self.stride = cfg.enc_stride
        self.residual = cfg.residual_mask

    def forward(self, mix, video=None, enroll=None):
        length = mix.shape[-1]
        enc = self.encoder(mix)
        sep = self.sepnet(enc)
        mask = self.scale * self.mask_head(sep)
        res = self.decoder(enc * mask)
        est = mix - res if self.residual else res
        if est.shape[-1] < length:
            est = F.pad(est, (0, length - est.shape[-1]))
        est = est[..., :length]
        return est, {"mask": mask}


class EnrollModel(nn.Module):
    """注册语音引导（VoiceFilter 风格）：d-vector -> FiLM 调制 TCN 特征

    与 AV 模型共享编码器 / TCN / 掩膜头规模，差别只在"线索来自哪"：
    AV 用口型视频，这里用同说话人的另一条语音。这是审稿人一定会要求的公平对照。
    """

    def __init__(self, cfg):
        super().__init__()
        N, C = cfg.n_filters, cfg.tcn_channels
        self.encoder = AudioEncoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.sepnet = SeparationNet(C, cfg.tcn_repeats, cfg.tcn_dilations)
        self.spk_enc = SpeakerEncoder(C)
        self.film_g = nn.Linear(C, C)
        self.film_b = nn.Linear(C, C)
        nn.init.zeros_(self.film_g.weight); nn.init.zeros_(self.film_g.bias)
        nn.init.zeros_(self.film_b.weight); nn.init.zeros_(self.film_b.bias)
        self.mask_head = MaskHead(2 * C, N, cfg.mask_act)
        self.decoder = AudioDecoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.scale = cfg.mask_scale
        self.residual = cfg.residual_mask

    def forward(self, mix, video=None, enroll=None):
        length = mix.shape[-1]
        e = self.spk_enc(enroll)                                  # [B, C]
        enc = self.encoder(mix)
        sep = self.sepnet(enc)
        g = 1.0 + self.film_g(e)[..., None]
        b = self.film_b(e)[..., None]
        sep = sep * g + b
        mask = self.scale * self.mask_head(torch.cat([sep, e[..., None].expand(-1, -1, sep.shape[-1])], dim=1))
        res = self.decoder(enc * mask)
        est = mix - res if self.residual else res
        if est.shape[-1] < length:
            est = F.pad(est, (0, length - est.shape[-1]))
        est = est[..., :length]
        return est, {"mask": mask}


class PITModel(nn.Module):
    """两输出、置换不变训练（PIT）的纯音频分离器 —— "分离能力"的上界基线

    与 AudioOnlyModel 唯一的区别是掩膜头输出 2N 通道、解码出两路波形。
    推理时用 oracle 选择（取两路中与目标更接近的一路），这样就把
    "能不能分开" 与 "知不知道哪个是目标" 两件事解耦：
        - au (单输出、无线索)      -> 目标身份不可知，退化为恒等映射
        - pit_oracle (双输出+选择) -> 分离能力上限，目标身份由 oracle 给定
        - av / enroll              -> 线索自己分辨目标身份
    """

    def __init__(self, cfg, n_src=2):
        super().__init__()
        N, C = cfg.n_filters, cfg.tcn_channels
        self.n_src = n_src
        self.encoder = AudioEncoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.sepnet = SeparationNet(C, cfg.tcn_repeats, cfg.tcn_dilations)
        self.mask_head = MaskHead(C, n_src * N, cfg.mask_act)
        self.decoder = AudioDecoder(N, cfg.enc_kernel, cfg.enc_stride)
        self.scale = cfg.mask_scale

    def _match(self, y, length):
        if y.shape[-1] < length:
            y = F.pad(y, (0, length - y.shape[-1]))
        return y[..., :length]

    def forward(self, mix, video=None, enroll=None):
        length = mix.shape[-1]
        enc = self.encoder(mix)
        sep = self.sepnet(enc)
        mask = self.scale * self.mask_head(sep)                  # [B, n_src*N, T']
        B, _, T = mask.shape
        mask = mask.view(B, self.n_src, -1, T)
        outs = [self._match(self.decoder(enc * mask[:, k]), length) for k in range(self.n_src)]
        est = outs[0]
        return est, {"mask": mask, "alt": outs[1]}


def build_model(kind, cfg):
    """kind: av | audio | enroll | av_noattn | av_nodistill | pit"""
    if kind == "av":
        return AttentiveAVModel(cfg)
    if kind == "av_noattn":
        return AttentiveAVModel(cfg, use_attn=False, use_aux=True)
    if kind == "av_nodistill":
        return AttentiveAVModel(cfg, use_attn=True, use_aux=False)
    if kind in ("audio", "audioonly", "baseline"):
        return AudioOnlyModel(cfg)
    if kind in ("enroll", "voicefilter"):
        return EnrollModel(cfg)
    if kind == "pit":
        return PITModel(cfg)
    raise ValueError(f"未知模型类型: {kind}")


NEEDS_ENROLL = {"enroll", "voicefilter", "e2e_enroll"}


def forward_batch(model, kind, mix, video=None, enroll=None):
    """统一前向入口：注册式模型多吃一路 enrollment"""
    if kind in NEEDS_ENROLL:
        return model(mix, video, enroll)
    return model(mix, video)
