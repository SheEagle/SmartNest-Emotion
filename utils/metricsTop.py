import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import r2_score

__all__ = ['MetricsTop']


class MetricsTop():
    def __init__(self, train_mode):
        if train_mode == "regression":
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_regression,
                'MOSEI': self.__eval_mosei_regression,
                'SIMS': self.__eval_sims_regression,
                'SIMS2': self.__eval_sims_regression,
                'SIMS3': self.__eval_sims_regression,
                'SIMS3L': self.__eval_sims_regression,
            }
        else:
            self.metrics_dict = {
                'MOSI': self.__eval_mosi_classification,
                'MOSEI': self.__eval_mosei_classification,
                'SIMS': self.__eval_sims_classification,
                'SIMS2': self.__eval_sims_classification,
                'SIMS3': self.__eval_sims_classification,
                'SIMS3L': self.__eval_sims_classification,
            }

    def __eval_mosi_classification(self, y_pred, y_true):
        """
        {
            "Negative": 0,
            "Neutral": 1,
            "Positive": 2   
        }
        """
        y_pred = y_pred.cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        # three classes
        y_pred_3 = np.argmax(y_pred, axis=1)
        Mult_acc_3 = accuracy_score(y_pred_3, y_true)
        F1_score_3 = f1_score(y_true, y_pred_3, average='weighted')
        # two classes 
        y_pred = np.array([[v[0], v[2]] for v in y_pred])
        # with 0 (<= 0 or > 0)
        y_pred_2 = np.argmax(y_pred, axis=1)
        y_true_2 = []
        for v in y_true:
            y_true_2.append(0 if v <= 1 else 1)
        y_true_2 = np.array(y_true_2)
        Has0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Has0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')
        # without 0 (< 0 or > 0)
        non_zeros = np.array([i for i, e in enumerate(y_true) if e != 1])
        y_pred_2 = y_pred[non_zeros]
        y_pred_2 = np.argmax(y_pred_2, axis=1)
        y_true_2 = y_true[non_zeros]
        Non0_acc_2 = accuracy_score(y_pred_2, y_true_2)
        Non0_F1_score = f1_score(y_true_2, y_pred_2, average='weighted')

        eval_results = {
            "Has0_acc_2": round(Has0_acc_2, 4),
            "Has0_F1_score": round(Has0_F1_score, 4),
            "Non0_acc_2": round(Non0_acc_2, 4),
            "Non0_F1_score": round(Non0_F1_score, 4),
            "Acc_3": round(Mult_acc_3, 4),
            "F1_score_3": round(F1_score_3, 4)
        }
        return eval_results

    def __eval_mosei_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __eval_sims_classification(self, y_pred, y_true):
        return self.__eval_mosi_classification(y_pred, y_true)

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def __eval_mosei_regression(self, y_pred, y_true, exclude_zero=False):
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
        test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)
        test_preds_a3 = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth_a3 = np.clip(test_truth, a_min=-1., a_max=1.)

        mae = np.mean(np.absolute(test_preds - test_truth))  # Average L1 distance between preds and truths
        corr = np.corrcoef(test_preds, test_truth)[0][1]
        mult_a7 = self.__multiclass_acc(test_preds_a7, test_truth_a7)
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        non_zeros_binary_truth = (test_truth[non_zeros] > 0)
        non_zeros_binary_preds = (test_preds[non_zeros] > 0)

        non_zeros_acc2 = accuracy_score(non_zeros_binary_preds, non_zeros_binary_truth)
        non_zeros_f1_score = f1_score(non_zeros_binary_truth, non_zeros_binary_preds, average='weighted')

        binary_truth = (test_truth >= 0)
        binary_preds = (test_preds >= 0)
        acc2 = accuracy_score(binary_preds, binary_truth)
        f_score = f1_score(binary_truth, binary_preds, average='weighted')

        eval_results = {
            "Has0_acc_2": round(acc2, 4),
            "Has0_F1_score": round(f_score, 4),
            "Non0_acc_2": round(non_zeros_acc2, 4),
            "Non0_F1_score": round(non_zeros_f1_score, 4),
            "Mult_acc_5": round(mult_a5, 4),
            "Mult_acc_7": round(mult_a7, 4),
            "MAE": round(mae, 4),
            "Corr": round(corr, 4)
        }
        return eval_results

    def __eval_mosi_regression(self, y_pred, y_true):
        return self.__eval_mosei_regression(y_pred, y_true)

    def __eval_sims_regression(self, y_pred, y_true):
        # 将预测和真实值转换为NumPy数组
        test_preds = y_pred.view(-1).cpu().detach().numpy()
        test_truth = y_true.view(-1).cpu().detach().numpy()

        # 对预测和真实值进行限制在[-1, 1]范围内
        test_preds = np.clip(test_preds, a_min=-1., a_max=1.)
        test_truth = np.clip(test_truth, a_min=-1., a_max=1.)

        # 将标签分为不同的类别
        # 1. 弱情感两类：[-0.6, 0.0] 和 (0.0, 0.6]
        ms_2 = [-1.01, 0.0, 1.01]
        weak_index_l = np.where(test_truth >= -0.4)[0]
        weak_index_r = np.where(test_truth <= 0.4)[0]
        weak_index = [x for x in weak_index_l if x in weak_index_r]
        test_preds_weak = test_preds[weak_index]
        test_truth_weak = test_truth[weak_index]
        test_preds_a2_weak = test_preds_weak.copy()
        test_truth_a2_weak = test_truth_weak.copy()
        for i in range(2):
            test_preds_a2_weak[np.logical_and(test_preds_weak > ms_2[i], test_preds_weak <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2_weak[np.logical_and(test_truth_weak > ms_2[i], test_truth_weak <= ms_2[i + 1])] = i

        # 2. 两类：[-1.0, 0.0] 和 (0.0, 1.0]
        ms_2 = [-1.01, 0.0, 1.01]
        test_preds_a2 = test_preds.copy()
        test_truth_a2 = test_truth.copy()
        for i in range(2):
            test_preds_a2[np.logical_and(test_preds > ms_2[i], test_preds <= ms_2[i + 1])] = i
        for i in range(2):
            test_truth_a2[np.logical_and(test_truth > ms_2[i], test_truth <= ms_2[i + 1])] = i

        # 3. 三类：[-1.0, -0.1], (-0.1, 0.1], (0.1, 1.0]
        ms_3 = [-1.01, -0.1, 0.1, 1.01]
        test_preds_a3 = test_preds.copy()
        test_truth_a3 = test_truth.copy()
        for i in range(3):
            test_preds_a3[np.logical_and(test_preds > ms_3[i], test_preds <= ms_3[i + 1])] = i
        for i in range(3):
            test_truth_a3[np.logical_and(test_truth > ms_3[i], test_truth <= ms_3[i + 1])] = i

        # 4. 五类：[-1.0, -0.7], (-0.7, -0.1], (-0.1, 0.1], (0.1, 0.7], (0.7, 1.0]
        ms_5 = [-1.01, -0.7, -0.1, 0.1, 0.7, 1.01]
        test_preds_a5 = test_preds.copy()
        test_truth_a5 = test_truth.copy()
        for i in range(5):
            test_preds_a5[np.logical_and(test_preds > ms_5[i], test_preds <= ms_5[i + 1])] = i
        for i in range(5):
            test_truth_a5[np.logical_and(test_truth > ms_5[i], test_truth <= ms_5[i + 1])] = i

        # 计算评估指标
        mae = np.mean(np.absolute(test_preds - test_truth))  # 预测和真实值之间的平均L1距离
        corr = np.corrcoef(test_preds, test_truth)[0][1]  # 相关系数
        mult_a2 = self.__multiclass_acc(test_preds_a2, test_truth_a2)  # 二分类准确率
        mult_a2_weak = self.__multiclass_acc(test_preds_a2_weak, test_truth_a2_weak)  # 弱情感二分类准确率
        mult_a3 = self.__multiclass_acc(test_preds_a3, test_truth_a3)  # 三分类准确率
        mult_a5 = self.__multiclass_acc(test_preds_a5, test_truth_a5)  # 五分类准确率
        f_score = f1_score(test_truth_a2, test_preds_a2, average='weighted')  # 加权F1分数
        r2 = r2_score(test_truth, test_preds)  # R平方值

        # 返回评估结果字典
        eval_results = {
            "Mult_acc_2": mult_a2,
            "Mult_acc_2_weak": mult_a2_weak,
            "Mult_acc_3": mult_a3,
            "Mult_acc_5": mult_a5,
            "F1_score": f_score,
            "MAE": mae,
            "Corr": corr,  # 相关系数
            "R_squre": r2
        }

        '''
        1. **Mult_acc_2 (二分类准确率):** 将预测的连续值映射到两个类别（例如，[-1.0, 0.0] 和 (0.0, 1.0]），然后计算准确率。

        2. **Mult_acc_2_weak (弱情感二分类准确率):** 将预测的连续值映射到两个弱情感类别（例如，[-0.6, 0.0] 和 (0.0, 0.6]），然后计算准确率。

        3. **Mult_acc_3 (三分类准确率):** 将预测的连续值映射到三个类别，然后计算准确率。

        4. **Mult_acc_5 (五分类准确率):** 将预测的连续值映射到五个类别，然后计算准确率。

        5. **F1_score (加权F1分数):** 计算二分类任务的F1分数，考虑到类别的不平衡。

        6. **MAE (平均绝对误差):** 计算预测值和真实值之间的平均绝对误差。

        7. **Corr (相关系数):** 计算预测值和真实值之间的相关系数。

        8. **R_squre (R平方值):** 衡量模型对总变异的解释程度，值越接近1表示模型拟合得越好。
        '''

        return eval_results

    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]
