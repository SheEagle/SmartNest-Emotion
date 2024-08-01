from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.subNets.BertTextEncoder import BertTextEncoder
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


__all__ = ['V1_Semi']


class SubNet(nn.Module):
    '''
    在 TFN 中用于音频和视频预融合阶段的子网络
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        参数:
            in_size: 输入维度
            hidden_size: 隐藏层维度
            dropout: 丢弃概率
        返回值:
            (forward 函数中的返回值) 形状为 (batch_size, hidden_size) 的张量
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)  # 批标准化层
        self.drop = nn.Dropout(p=dropout)  # 丢弃层
        self.linear_1 = nn.Linear(in_size, hidden_size)  # 全连接层1
        self.linear_2 = nn.Linear(hidden_size, hidden_size)  # 全连接层2
        self.linear_3 = nn.Linear(hidden_size, hidden_size)  # 全连接层3

    def forward(self, x):
        '''
        参数:
            x: 形状为 (batch_size, in_size) 的张量
        '''
        normed = self.norm(x)  # 批标准化
        dropped = self.drop(normed)  # 丢弃层
        y_1 = F.relu(self.linear_1(dropped))  # 全连接层1 + ReLU 激活函数
        y_2 = F.relu(self.linear_2(y_1))  # 全连接层2 + ReLU 激活函数
        y_3 = F.relu(self.linear_3(y_2))  # 全连接层3 + ReLU 激活函数
        return y_3


class AVsubNet(nn.Module):
    def __init__(self, in_size, hidden_size, dropout, bidirectional):
        super(AVsubNet, self).__init__()
        # 定义预融合子网络
        self.liner = nn.Linear(in_size, hidden_size)  # 全连接层
        self.dropout = nn.Dropout(dropout)  # 丢弃层
        self.rnn1 = nn.LSTM(hidden_size, hidden_size, bidirectional=bidirectional)  # LSTM层1
        self.rnn2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=bidirectional)  # LSTM层2
        self.layer_norm = nn.LayerNorm((2 * hidden_size,))  # Layer Normalization层

    def forward(self, sequence, lengths):
        lengths = lengths.squeeze().int().detach().cpu().view(-1)
        batch_size = sequence.shape[0]

        # sequence = self.dropout(self.liner(sequence))  # 全连接 + 丢弃层
        sequence = sequence.to(torch.float32)
        sequence = self.dropout(self.liner(sequence))  # 全连接 + 丢弃层
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = self.rnn1(packed_sequence)  # LSTM层1
        padded_h1, _ = pad_packed_sequence(packed_h1)
        padded_h1 = padded_h1.permute(1, 0, 2)
        normed_h1 = self.layer_norm(padded_h1)  # Layer Normalization
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = self.rnn2(packed_normed_h1)  # LSTM层2
        utterance = torch.cat((final_h1, final_h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return utterance


class Reconsitution(nn.Module):
    """模仿 ARGF 模型"""

    def __init__(self, args, input_dim, output_dim):
        super(Reconsitution, self).__init__()
        self.rec_dropout = nn.Dropout(args.rec_dropout)  # 丢弃层
        self.post_layer_1_rec = nn.Linear(input_dim, input_dim)  # 全连接层1
        self.post_layer_2_rec = nn.Linear(input_dim, output_dim)  # 全连接层2
        # self.tanh = nn.Tanh()

    def forward(self, input_feature):
        input_feature = self.rec_dropout(input_feature)  # 丢弃层
        input_feature1 = F.relu(self.post_layer_1_rec(input_feature))  # 全连接层1 + ReLU激活函数
        input_feature2 = self.post_layer_2_rec(input_feature1)  # 全连接层2
        return input_feature2


class V1_Semi(nn.Module):
    def __init__(self, args):
        super(V1_Semi, self).__init__()

        # 定义输入特征的维度
        self.text_in, self.audio_in, self.video_in = args.feature_dims

        # 定义隐藏层维度
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims

        # 定义丢弃率
        self.audio_prob, self.video_prob, self.text_prob = args.dropouts
        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts

        # 定义后处理层的维度
        self.post_fusion_dim = args.post_fusion_dim
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim

        # 文本模型（使用BERT进行编码）
        self.text_model = BertTextEncoder(language=args.language, use_finetune=args.use_bert_finetune,
                                          device=args.device)
        self.tliner = nn.Linear(self.text_in, self.text_hidden)

        # 音频和视频模型（使用AVsubNet进行处理）
        self.audio_model = AVsubNet(self.audio_in, self.audio_hidden, self.audio_prob, bidirectional=True)
        self.video_model = AVsubNet(self.video_in, self.video_hidden, self.video_prob, bidirectional=True)

        # 定义文本分类层
        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, 1)

        # 定义音频分类层
        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(4 * self.audio_hidden, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, 1)

        # 定义视频分类层
        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(4 * self.video_hidden, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, 1)

        # 定义融合分类层
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.post_text_dim + self.post_audio_dim + self.post_video_dim,
                                             self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, 1)

        # 重构层
        self.t_rec = Reconsitution(args, self.post_text_dim, self.text_in)
        self.a_rec = Reconsitution(args, self.post_audio_dim, self.audio_in)
        self.v_rec = Reconsitution(args, self.post_video_dim, self.video_in)

        # 输出范围和偏移
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def extract_features_eazy(self, audio, audio_lengths, vision, vision_lengths):
        vision_temp = []
        audio_temp = []

        # 提取音频和视频特征的简单方法
        for vi in range(len(vision_lengths)):
            vision_temp.append(torch.mean(vision[vi][:vision_lengths[vi]], axis=0))
        for ai in range(len(audio_lengths)):
            audio_temp.append(torch.mean(audio[ai][:audio_lengths[ai]], axis=0))
        vision_utt = torch.stack(vision_temp)
        audio_utt = torch.stack(audio_temp)
        return audio_utt, vision_utt

    def forward(self, text_x, audio_x, video_x):
        text_x, flag = text_x
        batch_size = text_x.shape[0]

        global text_h
        global audio_h
        global video_h

        # 根据标志位进行不同的数据处理
        if flag == 'train':
            # 数据预处理（对音频和视频进行处理）
            audio_x, a_len = audio_x
            video_x, v_len = video_x
            text_x = self.text_model(text_x)[:, 0, :]  # 使用文本模型提取特征
            text_h = self.tliner(text_x)  # 文本特征线性变换
            audio_h = self.audio_model(audio_x, a_len)  # 音频特征处理
            video_h = self.video_model(video_x, v_len)  # 视频特征处理

        if flag == 'mix_train':
            # 直接使用传入的特征，不进行额外处理
            text_h = text_x
            audio_h = audio_x
            video_h = video_x

        # 文本分类过程
        x_t1 = self.post_text_dropout(text_h)
        x_t2 = F.relu(self.post_text_layer_1(x_t1), inplace=True)
        x_t3 = F.relu(self.post_text_layer_2(x_t2), inplace=True)
        output_text = self.post_text_layer_3(x_t3)

        # 音频分类过程
        x_a1 = self.post_audio_dropout(audio_h)
        x_a2 = F.relu(self.post_audio_layer_1(x_a1), inplace=True)
        x_a3 = F.relu(self.post_audio_layer_2(x_a2), inplace=True)
        output_audio = self.post_audio_layer_3(x_a3)

        # 视频分类过程
        x_v1 = self.post_video_dropout(video_h)
        x_v2 = F.relu(self.post_video_layer_1(x_v1), inplace=True)
        x_v3 = F.relu(self.post_video_layer_2(x_v2), inplace=True)
        output_video = self.post_video_layer_3(x_v3)

        # 融合分类过程
        fusion_data = torch.cat([x_t2, x_a2, x_v2], dim=1)
        fusion_data = self.post_fusion_dropout(fusion_data)
        fusion_data = self.post_fusion_layer_1(fusion_data)
        fusion_data = self.post_fusion_layer_2(fusion_data)
        fusion_data = self.post_fusion_layer_3(fusion_data)

        output_fusion = torch.sigmoid(fusion_data)
        output_fusion = output_fusion * self.output_range + self.output_shift

        # 重构过程
        x_t2_rec = self.t_rec(x_t2)
        x_a2_rec = self.a_rec(x_a2)
        x_v2_rec = self.v_rec(x_v2)

        # 返回结果字典
        res = {
            'Feature_t': text_h,  # 文本特征
            'Feature_a': audio_h,  # 音频特征
            'Feature_v': video_h,  # 视频特征
            'M': output_fusion,  # 融合分类结果
            'T': output_text,  # 文本分类结果
            'A': output_audio,  # 音频分类结果
            'V': output_video  # 视频分类结果
        }

        return res
