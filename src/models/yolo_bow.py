from datetime import datetime
import pandas as pd

from src.core.device import Device
from src.core.model import Model
from src.core.pose import Pose
from src.core.video import Video
from src.core.log import logger

class YoloBow:
    @classmethod
    def process_video(cls, input_path, output_path, model_name='yolo11x-pose', device_name='auto', batch_size=12, bow_hand='left'):
        start_time = datetime.now()
        logger.info(f"▶️ 开始处理 {input_path} → {output_path}")

        device = Device.get_device(device_name)
        model = Model.get_model(model_name)
        model.to(device)
        logger.info(f"✅ 加载 {model.model_name} 模型到 {device} 设备")

        video = Video(input_path, output_path)
        # 数据记录 添加前手肩角
        records = pd.DataFrame(columns=['帧号', '双臂姿态角', '脊柱倾角', '前手肩角', '动作环节'])

        # 处理循环
        for processed, (frame, result) in enumerate(video.process_frames_batch(model, batch_size)):
            # 分析姿态
            frame, arm_angle, spine_angle, front_shoulder_angle, action_state = Pose.analyze_frame(frame, result, bow_hand)
            
            # 添加文本信息
            frame = Video.draw_texts(frame, (
                f"processed: {processed}", 
                f"Arm Angle: {arm_angle:.2f} deg",
                f"Spine Tilt: {spine_angle:.2f} deg",
                f"Front Shoulder: {front_shoulder_angle:.2f} deg",
                f"Technical process: {action_state.value}"
            ))

            # 记录数据
            records.loc[len(records)] = [processed, round(arm_angle, 2), round(spine_angle, 2), 
                                       round(front_shoulder_angle, 2), action_state.value]
            # 写入帧
            video.write_frame(frame)

        # 收尾工作
        video.close()
        # 创建CSV文件
        csv_path = output_path.rsplit('.', 1)[0] + '_data.csv'
        records.to_csv(csv_path, index=False, encoding='utf-8')

        total_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"✅ 处理完成: {processed}帧 | 总耗时 {total_time:.1f}s | "
            f"平均FPS {processed/total_time:.1f}\n"
            f"输出文件: {output_path}\n"
            f"数据文件: {csv_path}"
        )