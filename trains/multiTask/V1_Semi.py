import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop


logger = logging.getLogger('MSA')


class V1_Semi():
    def __init__(self, args):
        assert args.datasetName == 'sims3l'
        self.args = args
        self.args.tasks = "MTAV"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.recloss = nn.MSELoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        # 梯度下降的参数设置
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if
                              'text_model' not in n and 'audio_model' not in n and 'video_model' not in n]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert,
             'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other,
             'lr': self.args.learning_rate_other}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        # 初始化结果和计数
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        # 开始训练，直到满足早停条件
        while True:
            epochs += 1

            # 训练阶段
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0

            with tqdm(dataloader['train_mix']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    mask = batch_data['mask']
                    labels = batch_data['labels']

                    # 清零梯度
                    optimizer.zero_grad()
                    flag = 'train'

                    # 前向传播
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))

                    # 计算损失
                    loss = 0

                    # 1. 监督损失
                    labels_true = {}
                    outputs_true = {}
                    for k in labels.keys():
                        labels[k] = labels[k].to(self.args.device).view(-1, 1)
                        mask_index = torch.where(mask == 1)
                        labels_true[k] = labels[k][mask_index]
                        outputs_true[k] = outputs[k][mask_index]

                    for m in self.args.tasks:
                        if mask.sum() > 0:
                            loss += eval('self.args.' + m) * self.criterion(outputs_true[m], labels_true[m])

                    # 2. 无监督损失
                    flag = 'mix_train'
                    audio_utt = outputs['Feature_a']
                    prea = outputs['A']
                    video_utt = outputs['Feature_v']
                    prev = outputs['V']

                    loss_V_mix = 0.0
                    video_utt, video_utt_chaotic, video_utt_mix, y, y2, ymix, lam = mixup_data(video_utt, prev)
                    x_v1 = model.Model.post_video_dropout(video_utt_mix)
                    x_v2 = F.relu(model.Model.post_video_layer_1(x_v1), inplace=True)
                    x_v3 = F.relu(model.Model.post_video_layer_2(x_v2), inplace=True)
                    output_video = model.Model.post_video_layer_3(x_v3)
                    loss_V_mix += self.args.V * self.criterion(output_video, ymix)

                    loss_A_mix = 0.0
                    audio_utt, audio_utt_chaotic, audio_utt_mix, y, y2, ymix, lam = mixup_data(audio_utt, prea)
                    x_a1 = model.Model.post_audio_dropout(audio_utt_mix)
                    x_a2 = F.relu(model.Model.post_audio_layer_1(x_a1), inplace=True)
                    x_a3 = F.relu(model.Model.post_audio_layer_2(x_a2), inplace=True)
                    output_audio = model.Model.post_audio_layer_3(x_a3)
                    loss_A_mix += self.args.A * self.criterion(output_audio, ymix)

                    # 反向传播
                    loss += loss_A_mix
                    loss += loss_V_mix
                    if mask.sum() > 0:
                        loss.backward()
                        train_loss += loss.item()

                    # 更新参数
                    optimizer.step()

                    # 存储结果
                    for m in self.args.tasks:
                        y_pred[m].append(outputs_true[m].cpu())
                        y_true[m].append(labels_true['M'].cpu())

            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)

                    # 清零梯度
                    optimizer.zero_grad()
                    flag = 'train'

                    # 前向传播
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))

                    # 计算损失
                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.' + m) * self.criterion(outputs[m], labels[m])

                    # 反向传播
                    loss.backward()

                    # 更新参数
                    optimizer.step()

                    # 存储结果
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())

            train_loss = train_loss / len(dataloader['train_mix'])

            # 打印训练信息到控制台
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (
                self.args.modelName, epochs - best_epoch, epochs, self.args.cur_time,
                train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' % (m) + dict_to_str(train_results))

                # 打印训练信息到控制台
            print("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName,
                                                           epochs - best_epoch, epochs, self.args.cur_time,
                                                           train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                print('%s: >> ' % (m) + dict_to_str(train_results))

            # 验证
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]

            # 保存最佳模型
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            # 提前停止
            if epochs - best_epoch >= self.args.early_stop:
                return

    def do_test(self, model, dataloader, mode="VAL"):
        # 将模型设置为评估模式
        model.eval()
        # 初始化预测结果和真实标签的字典
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        # 初始化评估损失
        eval_loss = 0.0
        # 在没有梯度的情况下进行评估
        with torch.no_grad():
            # 使用 tqdm 显示进度条
            with tqdm(dataloader) as td:
                for batch_data in td:
                    # 获取批次数据
                    vision = batch_data['vision'].to(self.args.device)
                    print(vision.shape[0])
                    vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    # 计算每个视频的帧数
                    # vision_lengths_test = np.array([video.shape[0] for video in batch_data['vision_lengths']])
                    # print(vision_lengths_test)
                    print(vision_lengths)
                    audio = batch_data['audio'].to(self.args.device)
                    audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']
                    # 将标签移动到设备上
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)

                    # 设置标志为 'train'
                    flag = 'train'
                    # 获取模型输出
                    outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
                    # 初始化损失
                    loss = 0.0
                    # 计算总损失
                    for m in self.args.tasks:
                        loss += eval('self.args.' + m) * self.criterion(outputs[m], labels[m])
                    # 累加评估损失
                    eval_loss += loss.item()
                    # 将预测结果和真实标签添加到字典中
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())

        # 计算平均评估损失
        eval_loss = round(eval_loss / len(dataloader), 4)
        # 打印评估信息
        logger.info(mode + "-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        # 初始化评估结果字典
        eval_results = {}
        # 遍历任务，计算评估指标
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)
            logger.info('%s: >> ' % (m) + dict_to_str(results))
            eval_results[m] = results

        # 获取第一个任务的评估结果
        eval_results = eval_results[self.args.tasks[0]]
        # 添加损失到评估结果
        eval_results['Loss'] = eval_loss

        return eval_results

    def test_single(self, model, input, mode="VAL"):
        # 将模型设置为评估模式
        model.eval()
        # 初始化预测结果和真实标签的字典
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        # 初始化评估损失
        eval_loss = 0.0
        # 在没有梯度的情况下进行评估
        with torch.no_grad():
            vision = input['vision']
            vision = torch.from_numpy(vision)
            vision.to(self.args.device)

            vision_lengths = input['vision_lengths']
            vision_lengths = torch.from_numpy(vision_lengths)
            vision_lengths.to(self.args.device)

            audio = input['audio']
            audio = torch.from_numpy(audio)
            audio.to(self.args.device)

            audio_lengths = input['audio_lengths']
            audio_lengths = torch.from_numpy(audio_lengths)
            audio_lengths.to(self.args.device)

            text = input['text_bert']
            text = torch.from_numpy(text)
            text.to(self.args.device)

            # labels = input['labels']
            # 设置标志为 'train'
            flag = 'train'
            # 获取模型输出
            outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))
            for m in self.args.tasks:
                y_pred[m].append(outputs[m].cpu())
                # y_true[m].append(labels['M'].cpu())
        print(y_pred)
        return y_pred


def do_test_single(self, model, dataloader, mode="VAL"):
    # 将模型设置为评估模式
    model.eval()

    # 初始化预测结果和真实标签的字典
    y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
    y_true = {'M': [], 'T': [], 'A': [], 'V': []}

    # 初始化评估损失
    eval_loss = 0.0

    # 在没有梯度的情况下进行评估
    with torch.no_grad():
        # 使用 tqdm 显示进度条
        with tqdm(dataloader) as td:
            for batch_data in td:
                # 获取批次数据
                vision = batch_data['vision'].to(self.args.device)
                vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                audio = batch_data['audio'].to(self.args.device)
                audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                text = batch_data['text'].to(self.args.device)
                labels = batch_data['labels']

                # 将标签移动到设备上
                for k in labels.keys():
                    if self.args.train_mode == 'classification':
                        labels[k] = labels[k].to(self.args.device).view(-1).long()
                    else:
                        labels[k] = labels[k].to(self.args.device).view(-1, 1)

                # 设置标志为 'train'
                flag = 'train'

                # 获取模型输出
                outputs = model((text, flag), (audio, audio_lengths), (vision, vision_lengths))

                # 初始化损失
                loss = 0.0

                # 计算总损失
                for m in self.args.tasks:
                    loss += eval('self.args.' + m) * self.criterion(outputs[m], labels[m])

                # 累加评估损失
                eval_loss += loss.item()

                # 将预测结果和真实标签添加到字典中
                for m in self.args.tasks:
                    y_pred[m].append(outputs[m].cpu())
                    y_true[m].append(labels['M'].cpu())

    # 计算平均评估损失
    eval_loss = round(eval_loss / len(dataloader), 4)

    # 打印评估信息
    logger.info(mode + "-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)

    # 初始化评估结果字典
    eval_results = {}

    # 遍历任务，计算评估指标
    for m in self.args.tasks:
        pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
        results = self.metrics(pred, true)
        logger.info('%s: >> ' % (m) + dict_to_str(results))
        eval_results[m] = results

    # 获取第一个任务的评估结果
    eval_results = eval_results[self.args.tasks[0]]

    # 添加损失到评估结果
    eval_results['Loss'] = eval_loss

    return eval_results


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    x2 = x[index, :]
    y2 = y[index]
    xmix = lam * x + (1 - lam) * x2
    ymix = lam * y + (1 - lam) * y2
    y, y2 = y, y[index]
    return x, x2, xmix, y, y2, ymix, lam


def mixup_data_no_grad(x, y, y_m, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    y_m_a, y_m_b = y_m, y_m[index]
    return mixed_x, y_a, y_b, y_m_a, y_m_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
