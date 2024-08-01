# -*- codeing = utf-8 -*-
# @Time : 2024/2/5 13:52
# @Name : xiru wang
# @File : final_version.py
# @Software : PyCharm
import argparse
import datetime
import logging
import random
import numpy as np
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
from MSA_FET import FeatureExtractionTool
import os
import gc
import pickle
import torch
import pynvml
from models.AMIO import AMIO
from trains.ATIO import ATIO
from config.config_regression import ConfigRegression
import argparse
from mser.predict import MSERPredictor


def setup_seed(seed):
    # 设置PyTorch的随机种子
    torch.manual_seed(seed)
    # 设置CUDA的随机种子
    torch.cuda.manual_seed_all(seed)
    # 设置NumPy的随机种子
    np.random.seed(seed)
    # 设置Python内置random模块的随机种子
    random.seed(seed)
    # 设置cuDNN的随机种子，确保结果的确定性
    torch.backends.cudnn.deterministic = True


def extract_text_from_audio(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio, language='zh-CN')
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")


def extract_audio(input_video, output_audio):
    video_clip = VideoFileClip(input_video)
    audio_clip = video_clip.audio

    audio_clip.write_audiofile(output_audio)


def run_test(args, file_path):
    # 设置结果保存目录
    args.res_save_dir = os.path.join(args.res_save_dir, 'test')
    # 备份初始参数
    init_args = args
    # 存储模型结果
    # model_results = []
    seeds = args.seeds
    # 遍历不同的种子
    for i, seed in enumerate(seeds):
        args = init_args
        # 加载配置
        config = ConfigRegression(args)
        args = config.get_config()
        # 设置随机种子
        setup_seed(seed)
        args.seed = seed
        logger.info('开始运行 %s...' % (args.modelName))
        logger.info(args)
        # 运行
        args.cur_time = i + 1
        test_results = run_single_test(args, file_path)
        # 存储结果
        # model_results.append(test_results)
    return test_results


def run_single_test(args, file_path):
    # 如果模型保存目录不存在，则创建
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    # 指定模型保存路径
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.datasetName}-{args.seed}.pth')

    with open(file_path, 'rb') as f:
        input = pickle.load(f)

    # 指定使用的 GPU
    if len(args.gpu_ids) == 0 and torch.cuda.is_available():
        # 加载空闲内存最多的 GPU
        pynvml.nvmlInit()
        dst_gpu_id, min_mem_used = 0, 1e16
        for g_id in [0, 1, 2, 3]:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        print(f'Find gpu: {dst_gpu_id}, use memory: {min_mem_used}!')
        logger.info(f'Find gpu: {dst_gpu_id}, with memory: {min_mem_used} left!')
        args.gpu_ids.append(dst_gpu_id)

    # 设备选择
    using_cuda = len(args.gpu_ids) > 0 and torch.cuda.is_available()
    logger.info("Let's use %d GPUs!" % len(args.gpu_ids))
    device = torch.device('cuda:%d' % int(args.gpu_ids[0]) if using_cuda else 'cpu')
    args.device = device

    # 添加临时张量以增加 GPU 的临时内存占用
    tmp_tensor = torch.zeros((100, 100)).to(args.device)

    # 加载已经训练好的模型
    model = AMIO(args)
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    # 释放临时张量
    del tmp_tensor

    # 统计模型参数数量
    def count_parameters(model):
        answer = 0
        for p in model.parameters():
            if p.requires_grad:
                answer += p.numel()
        return answer

    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    # 获取训练器
    atio = ATIO().getTrain(args)

    results = atio.test_single(model, input, mode="TEST_FINAL")

    # 释放模型和 GPU 缓存
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return results


# def parse_args():
#     # 创建参数解析器
#     parser = argparse.ArgumentParser()
#
#     # 是否进行参数调整
#     parser.add_argument('--is_tune', type=bool, default=False, help='tune parameters ?')
#
#     # 是否进行测试
#     parser.add_argument('--is_test', type=bool, default=True, help='是否进行测试')
#
#     # 模型名称
#     parser.add_argument('--modelName', type=str, default='v1_semi', help='support v1/v1_semi')
#
#     # 数据集名称
#     parser.add_argument('--datasetName', type=str, default='sims3l', help='support sims3/sims3l')
#
#     # 训练模式，如回归/分类
#     parser.add_argument('--train_mode', type=str, default="regression", help='regression')
#
#     # 模型保存目录
#     parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save results.')
#
#     # 结果保存目录
#     parser.add_argument('--res_save_dir', type=str, default='results/baseline', help='path to save results.')
#
#     # 指定使用的 GPU
#     parser.add_argument('--gpu_ids', type=list, default=[0],
#                         help='indicates the gpus will be used. If none, the most-free gpu will be used!')
#
#     # 监督数据数量
#     parser.add_argument('--supvised_nums', type=int, default=2722, help='number of supervised data')
#
#     parser.add_argument('--configs', type=str, default='configs/bi_lstm.yml', help='配置文件')
#
#     # parser.add_argument('--audio_path', type=str, default='configs/bi_lstm.yml', help='配置文件')
#
#     parser.add_argument('--model_path', type=str, default='results/models/BidirectionalLSTM_CustomFeatures/best_model/',
#                         help='导出的预测模型文件路径')
#
#     # 解析 Flask 相关参数
#     # debug 模式
#     parser.add_argument('--debug', action='store_true', help='Enable debug mode')
#
#     # host
#     parser.add_argument('--host', type=str, default='127.0.0.1', help='Host IP address')
#
#     # port
#     parser.add_argument('--port', type=int, default=5000, help='Port number')
#
#     # 解析命令行参数
#     return parser.parse_args()

def parse_args():
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 是否进行参数调整
    parser.add_argument('--is_tune', type=bool, default=False, help='tune parameters ?')

    # 是否进行测试
    parser.add_argument('--is_test', type=bool, default=True, help='是否进行测试')

    # 模型名称
    parser.add_argument('--modelName', type=str, default='v1_semi', help='support v1/v1_semi')

    # 数据集名称
    parser.add_argument('--datasetName', type=str, default='sims3l', help='support sims3/sims3l')

    # 训练模式，如回归/分类
    parser.add_argument('--train_mode', type=str, default="regression", help='regression')

    # 模型保存目录
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save results.')

    # 结果保存目录
    parser.add_argument('--res_save_dir', type=str, default='results/baseline', help='path to save results.')

    # 指定使用的 GPU
    parser.add_argument('--gpu_ids', type=list, default=[0],
                        help='indicates the gpus will be used. If none, the most-free gpu will be used!')

    # 监督数据数量
    parser.add_argument('--supvised_nums', type=int, default=2722, help='number of supervised data')

    parser.add_argument('--configs', type=str, default='configs/bi_lstm.yml', help='配置文件')

    # parser.add_argument('--audio_path', type=str, default='configs/bi_lstm.yml', help='配置文件')

    parser.add_argument('--model_path', type=str, default='results/models/BidirectionalLSTM_CustomFeatures/best_model/',
                        help='导出的预测模型文件路径')

    # debug 模式
    debug = False

    # host
    host = '127.0.0.1'

    # port
    port = 5000

    # 返回参数命名空间对象
    return argparse.Namespace(is_tune=False, is_test=True, modelName='v1_semi', datasetName='sims3l',
                              train_mode='regression', model_save_dir='results/models', res_save_dir='results/baseline',
                              gpu_ids=[0], supvised_nums=2722, configs='configs/bi_lstm.yml',
                              model_path='results/models/BidirectionalLSTM_CustomFeatures/best_model/',
                              debug=debug, host=host, port=port)


def set_log(args):
    # 设置日志文件路径
    log_file_path = f'logs/{args.modelName}-{args.datasetName}.log'
    # 设置日志记录器的级别为INFO
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 移除已存在的所有处理器
    for ph in logger.handlers:
        logger.removeHandler(ph)

    # 添加FileHandler以记录日志到文件
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)

    return logger


def process_video(video_path):
    '''
    提取并保存音频(.wav)
    '''
    # 获取当前日期
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    base_audio_directory = 'audio'
    # 在保存目录后面加一层以日期命名的文件夹
    save_audio_directory = os.path.join(base_audio_directory, current_date)
    # 创建目录（如果不存在）
    os.makedirs(save_audio_directory, exist_ok=True)
    # 构建保存路径（日期+序号）
    file_number = 1
    audio_path = os.path.join(save_audio_directory, f"{file_number}.wav")
    # 确保路径不会覆盖已存在的文件
    while os.path.exists(audio_path):
        file_number += 1
        audio_path = os.path.join(save_audio_directory, f"{file_number}.wav")

    # 提取音频，转成wav
    extract_audio(video_path, audio_path)
    print("音频提取完毕：")

    '''
    提取并保存文字(.txt)
    '''
    text = extract_text_from_audio(audio_path)
    base_text_directory = 'text'
    # 在保存目录后面加一层以日期命名的文件夹
    save_text_directory = os.path.join(base_text_directory, current_date)
    # 创建目录（如果不存在）
    os.makedirs(save_text_directory, exist_ok=True)
    # 构建保存路径（日期+序号）
    file_number = 1
    text_path = os.path.join(save_text_directory, f"{file_number}.txt")
    # 确保路径不会覆盖已存在的文件
    while os.path.exists(text_path):
        file_number += 1
        text_path = os.path.join(save_text_directory, f"{file_number}.txt")
    # 将文本写入文件的最后一行
    with open(text_path, 'a', encoding='utf-8') as file:
        file.write(text + '\n')
    print("文本提取完毕：" + text)

    '''
    提取并保存特征(.pkl)
    '''
    fet = FeatureExtractionTool("custom_config.json")
    feature = fet.run_single(video_path, text=text)
    # 设置保存目录（相对路径）
    base_feature_directory = 'test_feature'
    # 在保存目录后面加一层以日期命名的文件夹
    save_feature_directory = os.path.join(base_feature_directory, current_date)
    # 创建目录（如果不存在）
    os.makedirs(save_feature_directory, exist_ok=True)
    # 构建保存路径（日期+序号）
    file_number = 1
    feature_path = os.path.join(save_feature_directory, f"{file_number}.pkl")
    # 确保路径不会覆盖已存在的文件
    while os.path.exists(feature_path):
        file_number += 1
        feature_path = os.path.join(save_feature_directory, f"{file_number}.pkl")
    # 保存提取的特征到文件
    with open(feature_path, 'wb') as f:
        pickle.dump(feature, f)
    print("特征提取完毕！")

    '''
    进行测试
    '''
    args = parse_args()
    # 设置日志记录器
    global logger
    logger = set_log(args)
    # 设置随机种子
    args.seeds = [1111]
    test_results = run_test(args, file_path=feature_path)

    # 获取识别器
    predictor = MSERPredictor(configs=args.configs,
                              model_path=args.model_path,
                              use_gpu=False)

    label, score = predictor.predict(audio_data=audio_path)

    '''
    保存结果，每天结果的保存在一个txt里
    '''
    save_result_directory = 'result'
    # 创建目录（如果不存在）
    os.makedirs(save_result_directory, exist_ok=True)
    # 构建保存路径
    result_path = os.path.join(save_result_directory, f"{current_date}.txt")
    # 检查文件是否存在
    file_exists = os.path.exists(result_path)
    with open(result_path, 'a', encoding='utf-8') as file:
        # 如果文件不存在，写入总 文字 声音 画面
        if not file_exists:
            file.write("总 文字 声音 画面 情绪类别\n")
        for key, value in test_results.items():
            tensor_value = round(value[0].item(), 4)  # 获取并保留四位小数
            file.write(str(tensor_value) + " ")  # 写入张量值到文件
        file.write(str(label))
        file.write("\n")  # 写入换行符

    print("结果录入完毕！")

# '''
# 提取并保存音频(.wav)
# '''
# # 要提取的 mp4 文件路径
# video_path = "0002.mp4"
# # 获取当前日期
# current_date = datetime.datetime.now().strftime("%Y-%m-%d")
# base_audio_directory = 'audio'
# # 在保存目录后面加一层以日期命名的文件夹
# save_audio_directory = os.path.join(base_audio_directory, current_date)
# # 创建目录（如果不存在）
# os.makedirs(save_audio_directory, exist_ok=True)
# # 构建保存路径（日期+序号）
# file_number = 1
# audio_path = os.path.join(save_audio_directory, f"{file_number}.wav")
# # 确保路径不会覆盖已存在的文件
# while os.path.exists(audio_path):
#     file_number += 1
#     audio_path = os.path.join(save_audio_directory, f"{file_number}.wav")
#
# # 提取音频，转成wav
# extract_audio(video_path, audio_path)
# print("音频提取完毕：")
#
# '''
# 提取并保存文字(.txt)
# '''
# text = extract_text_from_audio(audio_path)
# base_text_directory = 'text'
# # 在保存目录后面加一层以日期命名的文件夹
# save_text_directory = os.path.join(base_text_directory, current_date)
# # 创建目录（如果不存在）
# os.makedirs(save_text_directory, exist_ok=True)
# # 构建保存路径（日期+序号）
# file_number = 1
# text_path = os.path.join(save_text_directory, f"{file_number}.txt")
# # 确保路径不会覆盖已存在的文件
# while os.path.exists(text_path):
#     file_number += 1
#     text_path = os.path.join(save_text_directory, f"{file_number}.txt")
# # 将文本写入文件的最后一行
# with open(text_path, 'a', encoding='utf-8') as file:
#     file.write(text + '\n')
# print("文本提取完毕：" + text)
#
# '''
# 提取并保存特征(.pkl)
# '''
# fet = FeatureExtractionTool("custom_config.json")
# feature = fet.run_single(video_path, text=text)
# # 设置保存目录（相对路径）
# base_feature_directory = 'test_feature'
# # 在保存目录后面加一层以日期命名的文件夹
# save_feature_directory = os.path.join(base_feature_directory, current_date)
# # 创建目录（如果不存在）
# os.makedirs(save_feature_directory, exist_ok=True)
# # 构建保存路径（日期+序号）
# file_number = 1
# feature_path = os.path.join(save_feature_directory, f"{file_number}.pkl")
# # 确保路径不会覆盖已存在的文件
# while os.path.exists(feature_path):
#     file_number += 1
#     feature_path = os.path.join(save_feature_directory, f"{file_number}.pkl")
# # 保存提取的特征到文件
# with open(feature_path, 'wb') as f:
#     pickle.dump(feature, f)
# print("特征提取完毕！")
#
# '''
# 进行测试
# '''
# args = parse_args()
# # 设置日志记录器
# global logger
# logger = set_log(args)
# # 设置随机种子
# args.seeds = [1111]
# test_results = run_test(args, file_path=feature_path)
#
# # 获取识别器
# predictor = MSERPredictor(configs=args.configs,
#                           model_path=args.model_path,
#                           use_gpu=False)
#
# label, score = predictor.predict(audio_data=audio_path)
#
# '''
# 保存结果，每天结果的保存在一个txt里
# '''
# save_result_directory = 'result'
# # 创建目录（如果不存在）
# os.makedirs(save_result_directory, exist_ok=True)
# # 构建保存路径
# result_path = os.path.join(save_result_directory, f"{current_date}.txt")
# # 检查文件是否存在
# file_exists = os.path.exists(result_path)
# with open(result_path, 'a', encoding='utf-8') as file:
#     # 如果文件不存在，写入总 文字 声音 画面
#     if not file_exists:
#         file.write("总 文字 声音 画面 情绪类别\n")
#     for key, value in test_results.items():
#         tensor_value = round(value[0].item(), 4)  # 获取并保留四位小数
#         file.write(str(tensor_value) + " ")  # 写入张量值到文件
#     file.write(str(label))
#     file.write("\n")  # 写入换行符
#
# print("结果录入完毕！")
