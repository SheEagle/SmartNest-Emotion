# -*- codeing = utf-8 -*-
# @Time : 2024/2/26 16:44
# @Name : xiru wang
# @File : emotion_report.py
# @Software : PyCharm
import datetime
import os

import os
import datetime
from collections import Counter
import os
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen:
    def __init__(self, mode='offline', model_path="Qwen/Qwen-1_8B-Chat") -> None:
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        self.url = "http://ip:port"  # local server: http://ip:port
        self.headers = {
            "Content-Type": "application/json"
        }
        self.data = {
            "question": "北京有什么好玩的地方？"
        }
        self.prompt = '''请用少于25个字回答以下问题 '''
        self.mode = mode
        self.model, self.tokenizer = self.init_model(model_path)
        self.history = None

    def init_model(self, path="Qwen/Qwen-1_8B-Chat"):
        model = AutoModelForCausalLM.from_pretrained(path,
                                                     device_map="auto",
                                                     trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        return model, tokenizer

    def generate(self, question, system_prompt=""):
        if self.mode != 'api':
            self.data["question"] = self.prompt + question
            try:
                response, self.history = self.model.chat(self.tokenizer, self.data["question"], history=self.history,
                                                         system=system_prompt)
                # print(self.history)
                return response
            except Exception as e:
                print(e)
                return "对不起，你的请求出错了，请再次尝试。\nSorry, your request has encountered an error. Please try again.\n"
        else:
            return self.predict_api(question)

    def predict_api(self, question):
        '''暂时不写api版本,与Linly-api相类似,感兴趣可以实现一下'''
        pass

    def chat(self, system_prompt, message, history):
        response = self.generate(message, system_prompt)
        history.append((message, response))
        return response, history

    def clear_history(self):
        # 清空历史记录
        self.history = []


def read_emotion_data(filepath):
    emotion_data = {}

    # 从文件路径中提取日期
    date = os.path.splitext(os.path.basename(filepath))[0]

    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if lines:
            header = lines[0].strip()
            # values = [list(map(float, line.strip().split())) for line in lines[1:]]
            values = [list(map(float, line.strip().split()[:-1])) for line in lines[1:]]
            category = [line.strip().split()[-1] for line in lines[1:]]
            emotion_data[date] = {'header': header, 'values': values, 'category': category}

    return emotion_data


def classify_emotion(emotion_value):
    if emotion_value > 0.5:
        return '非常积极'
    elif 0 < emotion_value <= 0.5:
        return '比较积极'
    elif -0.5 <= emotion_value < 0:
        return '比较消极'
    else:
        return '非常消极'


def generate_daily_report(emotion_data):
    report_template = "用户的今日情绪：\n\n"
    for date, data in emotion_data.items():
        header = data['header']
        values = data['values']
        categorys = data['category']
        report_template += "{}:\n".format(date)

        # 统计每个分类的数量
        emotion_counts = {'非常积极': 0, '比较积极': 0, '比较消极': 0, '非常消极': 0}
        category_counts = {"中性": 0, "平静": 0, "快乐": 0, "悲伤": 0, "愤怒": 0, "恐惧": 0, "厌恶": 0, "惊讶": 0}
        for val in values:
            emotion = classify_emotion(val[0])
            emotion_counts[emotion] += 1

        # 情绪激烈的百分比
        total_emotion_count = len(values)
        for emotion, count in emotion_counts.items():
            percentage = (count / total_emotion_count) * 100
            report_template += "  - {}: {:.2f}%\n".format(emotion, percentage)

        # 具体情绪百分比
        for category in categorys:
            category_counts[category] += 1

        # 计算每个分类的百分比
        total_emotion_count = len(categorys)
        for emotion, count in category_counts.items():
            percentage = (count / total_emotion_count) * 100
            report_template += "  - {}: {:.2f}%\n".format(emotion, percentage)

    return report_template


# 示例使用
# result_directory = 'result'
# current_date = datetime.datetime.now().strftime("%Y-%m-%d")
# result_path = os.path.join(result_directory, "2024-02-26.txt")
# emotion_data = read_emotion_data(result_path)
# daily_report = generate_daily_report(emotion_data)
#
# print(daily_report)


