"""
AIO -- All Model in One
"""
import torch.nn as nn
from models.subNets.AlignNets import AlignSubNet
from models.multiTask import *

__all__ = ['AMIO']

MODEL_MAP = {
    # 多任务
    'v1': V1,
    'v1_semi': V1_Semi,
}


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.need_model_aligned = args.need_model_aligned
        # 模拟单词对齐网络（对于 seq_len_T == seq_len_A == seq_len_V）
        if self.need_model_aligned:
            self.alignNet = AlignSubNet(args, 'avg_pool')
            if 'seq_lens' in args.keys():
                args.seq_lens = self.alignNet.get_seq_len()
        lastModel = MODEL_MAP[args.modelName]
        self.Model = lastModel(args)

    def forward(self, text_x, audio_x, video_x):
        if self.need_model_aligned:
            text_x, audio_x, video_x = self.alignNet(text_x, audio_x, video_x)
        return self.Model(text_x, audio_x, video_x)
