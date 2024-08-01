import os
import logging
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

__all__ = ['MMDataLoader']
logger = logging.getLogger('MSA')


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'sims3l': self.__init_sims,
        }
        DATA_MAP[args.datasetName]()

    def __init_sims(self):
        # # 从文件中加载数据
        # with open(self.args.dataPath, 'rb') as f:
        #     data = pickle.load(f)
        #
        # train_mix = data['train']
        #
        # with open("D:\\智慧\\Emotion\\Emotion\\dataset\\unaligned_semi.pkl", 'rb') as f:
        #     train2 = pickle.load(f)
        #
        # # 遍历train2中的键值对，将数组添加到train中
        # for key, value in train2.items():
        #     if key in train_mix:
        #         # 如果键已经存在于train中，将数组追加到对应的值
        #         if np.issubdtype(value.dtype, np.floating):
        #             train_mix[key] = np.concatenate((train_mix[key], value), axis=0).astype(np.float32)
        #         else:
        #             train_mix[key] = np.concatenate((train_mix[key], value), axis=0)
        #
        # data['train_mix'] = train_mix
        #
        # # 保存数据集到文件
        # with open('D:\\智慧\\Emotion\\Emotion\\dataset\\data.pkl', 'wb') as f:
        #     pickle.dump(data, f)

        # 从文件加载数据集
        with open('/\\dataset\\data.pkl', 'rb') as f:
            data = pickle.load(f)

        # 控制有监督数据的数量
        if self.args.supvised_nums != 2722:
            # 在训练模式下，截取最后的有监督数据
            if self.mode == 'train':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    temp_data[self.mode][key] = data[self.mode][key][-self.args.supvised_nums:]
                data[self.mode] = temp_data[self.mode]

            # 在验证模式下，截取一半的有监督数据
            if self.mode == 'valid':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    p = int(self.args.supvised_nums / 2)
                    temp_data[self.mode][key] = data[self.mode][key][-p:]
                data[self.mode] = temp_data[self.mode]

            # 在混合训练模式下，合并有监督和无监督数据
            if self.mode == 'train_mix':
                temp_data = {}
                temp_data[self.mode] = {}
                for key in data[self.mode].keys():
                    data_sup = data[self.mode][key][2722 - self.args.supvised_nums:2722]
                    data_unsup = data[self.mode][key][2723:]
                    temp_data[self.mode][key] = np.concatenate((data_sup, data_unsup), axis=0)
                data[self.mode] = temp_data[self.mode]

        if self.args.train_mode == 'classification':
            data['train']['classification_labels'] = data['train']['annotations']
            data['valid']['classification_labels'] = data['valid']['annotations']
            data['test']['classification_labels'] = data['test']['annotations']
            data['train_mix']['classification_labels'] = data['train_mix']['annotations']
            # 创建LabelEncoder对象
            label_encoder = LabelEncoder()
            # 将标签编码为数字
            data['train']['classification_labels'] = label_encoder.fit_transform(data['train']['classification_labels'])
            print("标签到数字的映射：",
                  dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

            data['valid']['classification_labels'] = label_encoder.transform(data['valid']['classification_labels'])
            data['test']['classification_labels'] = label_encoder.transform(data['test']['classification_labels'])
            data['train_mix']['classification_labels'] = label_encoder.transform(
                data['train_mix']['classification_labels'])

        # 初始化数据
        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)
        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.ids = data[self.mode]['id']
        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']

        # 标签
        self.labels = {
            'M': data[self.mode][self.args.train_mode + '_labels'].astype(np.float32)
        }
        if self.args.datasetName == 'sims3l':
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode + '_labels_' + m]

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        # 在混合训练模式下，获取掩码数据
        if self.mode == 'train_mix':
            # self.mask = data[self.mode]['mask']
            self.mask = [1] * 2722 + [0] * 10161

        # 清理异常数据
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        # 如果需要标准化，执行标准化操作
        if self.args.need_normalized:
            self.__normalize()

    def __normalize(self):
        self.vision_temp = []
        self.audio_temp = []
        for vi in range(len(self.vision_lengths)):
            self.vision_temp.append(np.mean(self.vision[vi][:self.vision_lengths[vi]], axis=0))
        for ai in range(len(self.audio_lengths)):
            self.audio_temp.append(np.mean(self.audio[ai][:self.audio_lengths[ai]], axis=0))
        self.vision = np.array(self.vision_temp)
        self.audio = np.array(self.audio_temp)

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def __getitem__(self, index):
        # 返回一个样本
        sample = {
            'index': index,
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'audio_lengths': self.audio_lengths[index],
            'vision_lengths': self.vision_lengths[index],
            'mask': self.mask[index] if self.mode == 'train_mix' else [],
        }
        return sample


def MMDataLoader(args):
    # 创建包含不同数据集的字典
    datasets = {
        'train': MMDataset(args, mode='train'),  # 训练集
        'train_mix': MMDataset(args, mode='train_mix'),  # 混合训练集
        'valid': MMDataset(args, mode='valid'),  # 验证集
        'test': MMDataset(args, mode='test'),  # 测试集
    }

    # 如果参数中包含'seq_lens'，则将其设置为训练集的序列长度
    if 'seq_lens' in args:
        args.seq_lens = datasets['train'].get_seq_len()

    # 创建包含不同数据集 DataLoader 的字典
    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader
