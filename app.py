# -*- codeing = utf-8 -*-
# @Time : 2024/3/15 13:13
# @Name : xiru wang
# @File : app.py
# @Software : PyCharm
import datetime
import os

from flask import Flask, request, jsonify

from emotion_report import read_emotion_data, generate_daily_report, Qwen
from final_version import process_video

app = Flask(__name__)


def split_string_by_length(s, length):
    return [s[i:i + length] for i in range(0, len(s), length)]


@app.route('/get_report', methods=['GET'])
def get_report():
    result_directory = 'result'
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    result_path = os.path.join(result_directory, f"{current_date}.txt")
    emotion_data = read_emotion_data(result_path)
    report_template = generate_daily_report(emotion_data)
    report_template += '\n这是某位用户今天一天的情绪分析数据，其中包括了4种情绪的占比，' \
                       '请根据此数据生成一段情绪报告，' \
                       '报告中要包含针对性的情绪建议，' \
                       '口吻要面向用户，有亲和力'
    llm = Qwen(mode='offline', model_path="D:\\智慧\\Linly-Talker-main\\Linly-Talker-main\\Qwen\\Qwen\\Qwen-1_8B-Chat")
    answer = llm.generate(report_template)
    print(answer)

    # 将字符串每25个字符分成一行
    reports = split_string_by_length(answer, 25)

    # 返回分行后的字符串
    return jsonify({'report': reports})


def allowed_file(filename):
    # 检查文件扩展名是否在允许的视频格式中
    allowed_extensions = {'mp4'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/post_video', methods=['POST'])
def receive_video():
    # 检查是否有上传的文件
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400

    # 获取上传的视频文件
    video_file = request.files['video']

    # 检查文件是否为空
    if video_file.filename == '':
        return jsonify({'error': 'No selected video file'}), 400

    # 如果文件存在并且是视频文件
    if video_file and allowed_file(video_file.filename):
        # 生成随机文件名
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        base_video_directory = 'video'
        # 在保存目录后面加一层以日期命名的文件夹
        save_video_directory = os.path.join(base_video_directory, current_date)
        # 创建目录（如果不存在）
        os.makedirs(save_video_directory, exist_ok=True)
        # 构建保存路径（日期+序号）
        file_number = 1
        video_path = os.path.join(save_video_directory, f"{file_number}.mp4")
        # 确保路径不会覆盖已存在的文件
        while os.path.exists(video_path):
            file_number += 1
            video_path = os.path.join(save_video_directory, f"{file_number}.mp4")

        # 保存文件到服务器
        video_file.save(video_path)
        try:
            process_video(video_path)
        except Exception:
            print('情绪分析出现异常')
            return jsonify({'error': '情绪分析异常'}), 400
        else:
            return jsonify({'message': '视频上传成功'}), 200

    else:
        return jsonify({'error': 'Invalid video file format'}), 400


if __name__ == '__main__':
    app.run(debug=True)
