import os
import cv2
import logging
import math
import csv
from datetime import datetime

import torch
from ultralytics import YOLO
import numpy as np

from src.enums.action_state import ActionState

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()

class YoloBow:
    angle_list = []
    release_angle = None

    @classmethod
    def get_device(cls):
         # 自动选择最佳设备
        device = 'cuda' if torch.cuda.is_available() else \
                'mps' if torch.backends.mps.is_available() else 'cpu'
        if device == 'cuda':
            logger.info(f"🚀 使用CUDA加速: {torch.cuda.get_device_name(0)}")
        return device

    @classmethod
    def get_model(cls):
        # 初始化模型
        model_name = 'yolo11x-pose'
        model_path = f'data/models/{model_name}.pt'
        # 如果本地没有模型文件,则下载
        if not os.path.exists(model_path):
            logger.info(f"⏬ 下载 {model_name} 模型...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model = YOLO(f'{model_name}.pt')
            model.export(format='pt', file=model_path)  # 保存模型到本地
        else:
            logger.info(f"📂 使用本地 {model_name} 模型")
            model = YOLO(model_path)
        return model
    
    @classmethod
    def process_frames(cls, cap, model):
        # 定义帧缓冲区和批处理大小
        frame_buffer = []
        batch_size = 12  # 根据显存调整批处理大小
        while cap.isOpened():
            success, frame = cap.read()
            if not success: 
                if frame_buffer:
                    results = model.track(frame_buffer, imgsz=320, conf=0.5, verbose=False, stream=True)
                    for k, result in enumerate(results):
                        yield frame_buffer[k], result
                break
            # 将帧添加到缓冲区
            frame_buffer.append(frame)
            # 当缓冲区达到批处理大小时，进行批量处理
            if len(frame_buffer) == batch_size:
                # 批量处理帧
                results = model.track(frame_buffer, imgsz=320, conf=0.5, verbose=False, stream=True)
                # 处理结果（例如绘制轨迹等）
                for k, result in enumerate(results):
                    yield frame_buffer[k], result
                frame_buffer = []


    @classmethod
    def process_video(cls, input_path, output_path):
        start_time = datetime.now()

        # 创建CSV文件
        csv_path = output_path.rsplit('.', 1)[0] + '_data.csv'
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['帧号', '角度', '动作环节'])

        logger.info(f"▶️ 开始处理 {input_path} → {output_path}")

        device = cls.get_device()
        model = cls.get_model()
        model.to(device)
        logger.info(f"✅ 加载 {model.model_name} 模型到 {device} 设备")

        # 视频输入
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("❌ 无法打开视频文件")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        logger.info(f"📊 视频信息: {total_frames}帧 | {fps}FPS | 尺寸 {frame_size}")

        # 视频输出
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)

        # 处理循环
        processed = 0

        for frame, result in cls.process_frames(cap, model):
            frame = result.plot(boxes=False)
            angle = 0
            action_state = ActionState.UNKNOWN
            # 获取关键点数据
            keypoints = result.keypoints
            if keypoints is not None:
                for person in keypoints.xy:
                    if len(person) < 1:
                        continue
                    # 关键点顺序：鼻子、左眼、右眼、左耳、右耳、左肩、右肩、左肘、右肘、左腕、右腕、左髋、右髋、左膝、右膝、左脚踝、右脚踝
                    left_shoulder = person[5].cpu().numpy()
                    right_shoulder = person[6].cpu().numpy()
                    left_elbow = person[7].cpu().numpy()
                    right_elbow = person[8].cpu().numpy()
                    # todo 未完整识别到两臂坐标时不继续做分析处理，跳过进入下一帧
                    
                    # 绘制线段
                    cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_elbow[0]), int(left_elbow[1])), (0, 255, 0), 2)
                    cv2.line(frame, (int(right_shoulder[0]), int(right_shoulder[1])), (int(right_elbow[0]), int(right_elbow[1])), (0, 255, 0), 2)
                    # 计算夹角
                    angle = cls.calculate_angle(left_shoulder, left_elbow, right_shoulder, right_elbow)
                    # 获取动作环节
                    action_state = cls.judge_action(angle)
                    # 记录数据到CSV
                    csv_writer.writerow([processed, f"{angle:.2f}", action_state.value])
                    # 绘制角度值
                    cv2.putText(frame, f"Angle: {angle:.2f} deg", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # 绘制技术环节
                    cv2.putText(frame, f"Technical process: {action_state.value} ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    # 绘制帧序号
                    cv2.putText(frame, f"processed: {processed} ", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            writer.write(frame)

            # 进度日志
            processed += 1
            if processed % 30 == 0:  # 每30帧输出一次进度
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_log = processed / elapsed
                remain = (total_frames - processed) / fps_log if fps_log > 0 else 0
                logger.info(
                    f"⏳ 进度: {processed}/{total_frames} "
                    f"({processed/total_frames:.0%}) | "
                    f"耗时: {elapsed:.1f}s | "
                    f"剩余: {remain:.1f}s"
                )

        # 收尾工作
        cap.release()
        writer.release()
        csv_file.close()

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 处理完成: {processed}帧 | 总耗时 {total_time:.1f}s | "
            f"平均FPS {processed/total_time:.1f}\n"
            f"输出文件: {output_path}\n"
            f"数据文件: {csv_path}"
        )

    @staticmethod
    def calculate_angle(c, d, a, b) -> float:
        """计算两向量夹角（0-360度）"""
        # 转换为numpy数组
        vec_ab = np.array([b[0]-a[0], b[1]-a[1]])
        vec_cd = np.array([d[0]-c[0], d[1]-c[1]])
        
        # 计算模长
        norm_ab = np.linalg.norm(vec_ab)
        norm_cd = np.linalg.norm(vec_cd)
        
        if norm_ab == 0 or norm_cd == 0:
            return 0.0
            
        # 计算夹角（带方向）
        cos_theta = np.dot(vec_ab, vec_cd) / (norm_ab * norm_cd)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        
        # 判断方向
        cross = np.cross(vec_ab, vec_cd)
        angle_deg = np.degrees(angle_rad)
        return angle_deg if cross >= 0 else 360 - angle_deg

    @classmethod
    def judge_action(cls, angle):
        """
        根据角度判断动作环节
        参数:
            angle (float): 计算出的角度值 (0-360范围)
        """
        cls.angle_list.append(angle)
        
        release_angle_threshold = 4.5  # 固势->撒放 角度骤增差值阈值

        if 330 <= angle < 360 or 0 < angle < 12:
            cls.release_angle = None  # 重置撒放角
            return ActionState.LIFT  # 举弓
        elif 12 <= angle < 150:
            return ActionState.DRAW  # 开弓
        elif cls.release_angle and cls.release_angle - release_angle_threshold <= angle <= 185:
            return ActionState.RELEASE  # 撒放
        elif 150 <= angle < 185:
            previous_angles = cls.angle_list[-4:-1]
            previous_angle = sum(previous_angles) / 3  # 取前三帧的平均值
            if min(previous_angles) >= 150 and 20 > angle - previous_angle >= release_angle_threshold:  # 固势下骤增角度可视为进入撒发环节 (撒放角)
                cls.release_angle = angle
                return ActionState.RELEASE  # 撒放
            return ActionState.SOLID  # 固势
        elif 185 <= angle < 215:
            return ActionState.RELEASE  # 撒放
        else:
            return ActionState.UNKNOWN
