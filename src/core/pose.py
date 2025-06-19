import numpy as np
import cv2

from src.enums.action_state import ActionState

class Pose:
    
    angle_list = []
    release_angle = None
    
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
    def judge_action(cls, arm_angle):
        """
        根据双臂姿态角判断动作环节
        参数:
            arm_angle (float): 计算出的双臂姿态角 (0-360范围)
        """
        cls.angle_list.append(arm_angle)
        
        release_angle_threshold = 4.5  # 固势->撒放 角度骤增差值阈值

        if 330 <= arm_angle < 360 or 0 < arm_angle < 12:
            cls.release_angle = None  # 重置撒放角
            return ActionState.LIFT  # 举弓
        elif 12 <= arm_angle < 150:
            return ActionState.DRAW  # 开弓
        elif cls.release_angle and cls.release_angle - release_angle_threshold <= arm_angle <= 185:
            return ActionState.RELEASE  # 撒放
        elif 150 <= arm_angle < 185:
            previous_angles = cls.angle_list[-4:-1]
            previous_angle = sum(previous_angles) / 3  # 取前三帧的平均值
            if min(previous_angles) >= 150 and 20 > arm_angle - previous_angle >= release_angle_threshold:  # 固势下骤增角度可视为进入撒发环节 (撒放角)
                cls.release_angle = arm_angle
                return ActionState.RELEASE  # 撒放
            return ActionState.SOLID  # 固势
        elif 185 <= arm_angle < 215:
            return ActionState.RELEASE  # 撒放
        else:
            return ActionState.UNKNOWN

    @classmethod
    def analyze_frame(cls, frame, result, bow_hand):
        """分析单帧中的姿态数据"""
        frame = result.plot(boxes=False)
        arm_angle = 0
        spine_angle = 0
        front_shoulder_angle = 0  # 添加前手肩角
        action_state = ActionState.UNKNOWN

        keypoints = result.keypoints
        if keypoints is not None:
            for person in keypoints.xy:
                if len(person) < 1:
                    continue
                # 提取关键点数据
                left_shoulder = person[5].cpu().numpy()
                right_shoulder = person[6].cpu().numpy()
                left_elbow = person[7].cpu().numpy()
                right_elbow = person[8].cpu().numpy()
                left_hip = person[11].cpu().numpy()
                right_hip = person[12].cpu().numpy()

                # 计算关键点
                shoulder_midpoint = (left_shoulder + right_shoulder) / 2
                hip_midpoint = (left_hip + right_hip) / 2
                
                # 计算脊柱倾角
                spine_vector = shoulder_midpoint - hip_midpoint
                vertical_vector = np.array([0, -1])
                spine_angle = cls.calculate_angle(hip_midpoint, shoulder_midpoint, hip_midpoint, hip_midpoint + vertical_vector)
                if spine_angle > 180:
                    spine_angle = spine_angle - 360

                # 根据持弓手判断前手
                if bow_hand == 'left':
                    front_elbow = right_elbow
                    front_shoulder = right_shoulder
                    front_hip = right_hip
                else:
                    front_elbow = left_elbow
                    front_shoulder = left_shoulder
                    front_hip = left_hip
                # 计算前手肩角（肘关节中心→肩关节中心→髋关节中心）
                front_shoulder_angle = cls.calculate_angle(front_shoulder, front_elbow, front_shoulder, front_hip)

                # 绘制脊柱线段和前手肩部连线
                cls.draw_line(frame, hip_midpoint, shoulder_midpoint)
                cls.draw_line(frame, front_elbow, front_shoulder, color=(0, 255, 0))  # 绿色
                cls.draw_line(frame, front_shoulder, front_hip, color=(0, 255, 0))    # 绿色

                # 计算双臂姿态角并判断动作环节
                arm_angle = cls.calculate_angle(left_shoulder, left_elbow, right_shoulder, right_elbow)
                action_state = cls.judge_action(arm_angle)

        return frame, arm_angle, spine_angle, front_shoulder_angle, action_state

    @staticmethod
    def draw_line(frame, point_1, point_2, color=(255, 0, 0)):
        """绘制线段，允许自定义颜色"""
        cv2.line(frame, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])), color, 2)