"""
AIO -- All Trains in One
"""

from trains.multiTask import *

# 导出的模块列表
__all__ = ['ATIO']


class ATIO():
    def __init__(self):
        # 训练模型映射字典
        self.TRAIN_MAP = {
            # 多任务
            'v1': V1,
            'v1_semi': V1_Semi,
        }

    def getTrain(self, args):
        # 根据传入的模型名称返回对应的训练模型类实例
        return self.TRAIN_MAP[args.modelName.lower()](args)
