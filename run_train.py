import os
import gc
import pickle
import time
import random
import logging
import torch
import pynvml
import argparse
import numpy as np
import pandas as pd

from models.AMIO import AMIO
from trains.ATIO import ATIO
from data.load_data import MMDataLoader
from config.config_tune import ConfigTune
from config.config_regression import ConfigRegression

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


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


def run(args):
    # 如果模型保存目录不存在，则创建
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    # 指定模型保存路径
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.modelName}-{args.datasetName}-{args.seed}.pth')

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

    # 加载数据和模型
    dataloader = MMDataLoader(args)
    model = AMIO(args)
    # assert os.path.exists(args.model_save_path)
    # model.load_state_dict(torch.load(args.model_save_path))
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

    # 进行训练
    # atio.do_train(model, dataloader)

    # 加载预训练模型
    assert os.path.exists(args.model_save_path)
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(device)

    # 进行测试
    if args.is_tune:
        # 使用验证集调整超参数
        results = atio.do_test(model, dataloader['test'], mode="TEST")
    else:
        # 在测试集上进行最终测试
        results = atio.do_test(model, dataloader['test'], mode="TEST_FINAL")

    # 释放模型和 GPU 缓存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return results


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
    time.sleep(1)

    return results


def run_tune(args, tune_times=50):
    args.res_save_dir = os.path.join(args.res_save_dir, 'tunes')
    init_args = args
    has_debuged = []  # save used paras
    save_file_path = os.path.join(args.res_save_dir, \
                                  f'{args.datasetName}-{args.modelName}-tune.csv')
    if not os.path.exists(os.path.dirname(save_file_path)):
        os.makedirs(os.path.dirname(save_file_path))

    for i in range(tune_times):
        # cancel random seed
        setup_seed(int(time.time()))
        args = init_args
        config = ConfigTune(args)
        args = config.get_config()
        print(args)
        # print debugging params
        logger.info("#" * 40 + '%s-(%d/%d)' % (args.modelName, i + 1, tune_times) + '#' * 40)
        for k, v in args.items():
            if k in args.d_paras:
                logger.info(k + ':' + str(v))
        logger.info("#" * 90)
        logger.info('Start running %s...' % (args.modelName))
        # restore existed paras
        if i == 0 and os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
            for i in range(len(df)):
                has_debuged.append([df.loc[i, k] for k in args.d_paras])
        # check paras
        cur_paras = [args[v] for v in args.d_paras]
        if cur_paras in has_debuged:
            logger.info('These paras have been used!')
            time.sleep(3)
            continue
        has_debuged.append(cur_paras)
        results = []
        for j, seed in enumerate([1111, 1112, 1113]):
            args.cur_time = j + 1
            setup_seed(seed)
            results.append(run(args))
        # save results to csv
        logger.info('Start saving results...')
        if os.path.exists(save_file_path):
            df = pd.read_csv(save_file_path)
        else:
            df = pd.DataFrame(columns=[k for k in args.d_paras] + [k for k in results[0].keys()])
        # stat results
        tmp = [args[c] for c in args.d_paras]
        for col in results[0].keys():
            values = [r[col] for r in results]
            tmp.append(round(sum(values) * 100 / len(values), 2))

        df.loc[len(df)] = tmp
        df.to_csv(save_file_path, index=None)
        logger.info('Results are saved to %s...' % (save_file_path))


def run_normal(args):
    # 设置结果保存目录
    args.res_save_dir = os.path.join(args.res_save_dir, 'normals')
    # 备份初始参数
    init_args = args
    # 存储模型结果
    model_results = []
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
        test_results = run(args)
        # 存储结果
        model_results.append(test_results)
    # 获取评价指标列表
    criterions = list(model_results[0].keys())
    # 加载其他结果
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # 存储结果到DataFrame
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        # 计算均值和标准差并保留两位小数
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    # 将结果添加到DataFrame并保存
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('结果已添加到 %s...' % (save_path))


def run_test(args, file_path):
    # 设置结果保存目录
    args.res_save_dir = os.path.join(args.res_save_dir, 'test')
    # 备份初始参数
    init_args = args
    # 存储模型结果
    model_results = []
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
        model_results.append(test_results)
    # 获取评价指标列表
    criterions = list(model_results[0].keys())
    # 加载其他结果
    save_path = os.path.join(args.res_save_dir, f'{args.datasetName}.csv')
    if not os.path.exists(args.res_save_dir):
        os.makedirs(args.res_save_dir)
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    # 存储结果到DataFrame
    res = [args.modelName]
    for c in criterions:
        values = [r[c] for r in model_results]
        # 计算均值和标准差并保留两位小数
        mean = round(np.mean(values) * 100, 2)
        std = round(np.std(values) * 100, 2)
        res.append((mean, std))
    # 将结果添加到DataFrame并保存
    df.loc[len(df)] = res
    df.to_csv(save_path, index=None)
    logger.info('结果已添加到 %s...' % (save_path))


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


def parse_args():
    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 是否进行参数调整
    parser.add_argument('--is_tune', type=bool, default=False, help='tune parameters ?')

    # 是否进行测试
    parser.add_argument('--is_test', type=bool, default=False, help='是否进行测试')

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

    # 解析命令行参数
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    # 设置日志记录器
    global logger
    logger = set_log(args)

    # 设置随机种子
    args.seeds = [1111]

    # 如果是进行调参
    if args.is_tune:
        run_tune(args, tune_times=50)
    else:
        # 测试
        if args.is_test:
            run_test(args, file_path="test_feature/feature1.pkl")
            # run_single_test(args, file_path="test_feature/feature1.pkl")
        # 否则进行正常运行
        else:
            run_normal(args)

    '''
    M: 代表融合后的结果，表示文本、音频和视频信息的整合。
    T: 代表文本方面的结果
    A: 代表音频方面的结果
    V: 代表视频方面的结果
    回归分数：正数是positive，负数是negative，0是中性，绝对值越大情绪越重
    '''
