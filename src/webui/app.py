import gradio as gr
import os
import pandas as pd
import cv2
from src.models.yolo_bow import YoloBow

def process_video(video_path):
    """处理上传的视频文件"""
    if not video_path:
        return None
    
    try:
        # 获取输入视频的文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # 创建输出目录（如果不存在）
        output_dir = os.path.join("data", "output")
        os.makedirs(output_dir, exist_ok=True)
        # 构建输出文件路径
        output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
        csv_path = os.path.join(output_dir, f"{base_name}_processed_data.csv")
        # 处理视频
        YoloBow.process_video(video_path, output_path)
        
        # 读取CSV数据
        angles = pd.read_csv(csv_path, encoding='utf8')
        angles['定位'] = ''
        return {
            "video_path": output_path,
            "angles": angles
        }
    except Exception as e:
        print(f"处理视频时发生错误: {str(e)}")
        raise e
        return None            
    

def extract_frame(video_path, frame_number):
    """从视频中提取指定帧号的图像"""
    if not video_path or not os.path.exists(video_path):
        return None
    
    try:
        cap = cv2.VideoCapture(video_path)
        # 设置帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # 读取指定帧
        success, frame = cap.read()
        cap.release()
        
        if success:
            # 将BGR格式转换为RGB格式
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            print(f"无法读取帧 {frame_number}")
            return None
    except Exception as e:
        print(f"提取帧时发生错误: {str(e)}")
        return None

def process_with_status(video):
    if not video:
        return None, None, "请先上传视频", None, None
    result = process_video(video)
    if result:
        # 准备折线图数据(转换为DataFrame)
        slider = gr.Slider(minimum=0, maximum=len(result["angles"]), value=5, step=1, label="拖动滑块移动游标", interactive=True)
        # 提取第一帧作为初始帧
        initial_frame = extract_frame(result["video_path"], 5)
        return result["video_path"], result["angles"], "处理完成", slider, initial_frame
    else:
        return None, None, "处理失败，请检查控制台输出", None, None


# 视频播放时更新游标线
def update_cursor(data, slider):
    if data:
        data['data'] = [point for point in data['data'] if point[3] != '游标']  # 删除之前的游标数据
        data['data'].extend([[slider, 0, '', '游标'],[slider, 360, '', '游标'],])  # 添加新的游标数据
        return data
    
    return None

# 滑动滑块时提取并显示对应的视频帧
def update_frame(video_path, frame_number):
    if not video_path:
        return None
    
    frame = extract_frame(video_path, frame_number)
    return frame


# todo 合并 process_video + process_with_status

# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="Archery Vision", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎯 Archery Vision")
        
        arm_plot_data = gr.State() 
        # todo 模型选择
        # todo 选择 batch size
        # todo 左右手持弓选择，默认左手持弓
        with gr.Row():
            with gr.Column():
                input_video = gr.Video(label="上传视频", sources="upload", interactive=True)
                process_btn = gr.Button("开始分析", variant="primary")
            with gr.Column():
                output_video = gr.Video(label="分析结果", format="mp4", interactive=False)
            with gr.Column():            
                status_text = gr.Textbox(label="处理状态", interactive=False, value="等待上传视频...")
        with gr.Row():
            current_frame = gr.Image(label="当前帧", type="numpy", interactive=False)
        with gr.Row():
            # slider = gr.Slider(0, 10, value=5, step=0.1, label="拖动滑块移动游标")
            slider = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="拖动滑块移动游标", interactive=True)
        # 添加姿态角折线图
        with gr.Row():
            arm_plot = gr.LinePlot(label="双臂姿态角", x="帧号", y="角度", color='定位', width=500, height=300)
            
        slider.change(
            fn=update_cursor,
            inputs=[arm_plot, slider],
            outputs=[arm_plot]
        )
        
        # 添加滑块改变时更新帧图像的事件
        slider.change(
            fn=update_frame,
            inputs=[output_video, slider],
            outputs=[current_frame]
        )

        process_btn.click(
            fn=process_with_status,
            inputs=[input_video],
            outputs=[output_video, arm_plot, status_text, slider, current_frame]
        )
        
    return app

if __name__ == "__main__":
    # 创建并启动UI
    app = create_ui()
    app.queue()  # 启用队列处理以提高稳定性
    app.launch(
        server_name="127.0.0.1",  # 只监听本地连接
        show_error=True,
        quiet=True,  # 减少控制台输出
        share=False
    )
