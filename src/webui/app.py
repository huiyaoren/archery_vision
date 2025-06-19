import gradio as gr
import os
import pandas as pd
from src.core.video import Video
from src.models.yolo_bow import YoloBow


def process_video(video_path, user_options):
    """处理上传的视频文件"""
    if not video_path:
        return "请先上传视频", *[None]*8  # 修改为8个None，总共9个返回值

    # 获取输入视频的文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 创建输出目录（如果不存在）
    output_dir = os.path.join("data", "output")
    os.makedirs(output_dir, exist_ok=True)
    # 构建输出文件路径
    output_path = os.path.join(output_dir, f"{base_name}_processed.mp4")
    csv_path = os.path.join(output_dir, f"{base_name}_processed_data.csv")
    # 处理视频
    YoloBow.process_video(video_path, output_path, 
                            model_name=user_options.get('model_dropdown', 'yolo11x-pose'), 
                            device_name=user_options.get('device_dropdown', 'auto'),
                            batch_size=user_options.get('batch_size', 8))
    # 读取CSV数据
    angles = pd.read_csv(csv_path, encoding='utf8')
    
    # 计算角速度，处理0-360度切换
    def calculate_angle_diff(angles_series):
        diff = angles_series.diff()
        # 处理角度突变
        mask = abs(diff) > 180
        diff.loc[mask & (diff > 0)] -= 360
        diff.loc[mask & (diff < 0)] += 360
        return diff

    # 计算双臂姿态角速度 (度/帧)
    angles['角速度'] = calculate_angle_diff(angles['双臂姿态角'])
    
    # 计算角加速度 (度/帧^2)
    angles['角加速度'] = angles['角速度'].diff()
    
    # 准备不同图表的数据
    arm_angle_data = angles  # 双臂姿态角数据
    
    # 添加脊柱倾角警告标记
    spine_angle_data = angles[['帧号', '脊柱倾角']].copy()
    spine_angle_data['警告'] = spine_angle_data['脊柱倾角'].apply(lambda x: '角度')

    # 添加参考线数据
    reference_data = pd.DataFrame({
        '帧号': [angles['帧号'].min(), angles['帧号'].max()],
        '+5°警戒线': [5, 5],
        '-5°警戒线': [-5, -5]
    })
    spine_angle_data = pd.concat([
        reference_data[['帧号', '+5°警戒线']].rename(columns={'+5°警戒线': '脊柱倾角'}).assign(警告='+5°警戒线'),
        reference_data[['帧号', '-5°警戒线']].rename(columns={'-5°警戒线': '脊柱倾角'}).assign(警告='-5°警戒线'),
        spine_angle_data
    ])

    velocity_data = angles[['帧号', '角速度']]  # 角速度数据
    acceleration_data = angles[['帧号', '角加速度']]  # 角加速度数据
    phase_data = angles[['双臂姿态角', '角速度']]  # 相位图数据

    # 准备折线图数据
    slider = gr.Slider(minimum=0, maximum=len(angles), value=5, step=1, label="拖动滑块移动游标", interactive=True)
    initial_frame = Video.extract_frame(output_path, 5)

    return ("处理完成", output_path, slider, initial_frame, arm_angle_data, spine_angle_data, velocity_data, acceleration_data, phase_data)


# 视频播放时更新游标线
def update_cursor(data, slider):
    if data:
        return data
    
    return None


# 创建Gradio界面
def create_ui():
    with gr.Blocks(title="Archery Vision", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎯 Archery Vision")
        
        with gr.Tabs() as tabs:

            with gr.Tab("1.视频处理"):
                with gr.Row():
                    user_options = gr.BrowserState({})
                    device_dropdown = gr.Dropdown( label="设备选择", choices=["auto", "cpu", "cuda", "mps"], value="auto", interactive=True)
                    bow_hand = gr.Dropdown( label="持弓手", choices=["left", "right"], value="left", interactive=True)
                    model_dropdown = gr.Dropdown( label="模型选择", choices=["yolov8x-pose-p6", "yolo11x-pose"], value="yolov8x-pose-p6", interactive=True)
                    batch_size = gr.Number(label="Batch Size", value=8, minimum=1, maximum=64, step=2, precision=0, interactive=True)
                with gr.Row():
                    with gr.Column():
                        input_video = gr.Video(label="上传视频", sources="upload", interactive=True)
                    with gr.Column():
                        output_video = gr.Video(label="分析结果", format="mp4", interactive=False)
                with gr.Row():         
                    process_btn = gr.Button("开始分析", variant="primary")
                with gr.Row():         
                    status_text = gr.Textbox(label="处理状态", interactive=False, value="等待上传视频...")

            with gr.Tab("2.数据分析"):
                with gr.Row():
                    with gr.Column(scale=4):
                        current_frame = gr.Image(label="当前帧", type="numpy", interactive=False)
                    with gr.Column(scale=1):
                        current_frame_data = gr.Dataframe(headers=["指标", "数值"], label="当前帧数据", interactive=False)
                        refresh_btn = gr.Button("更新图表", variant="secondary")
                with gr.Row():
                    slider = gr.Slider(minimum=0, maximum=100, value=5, step=1, label="拖动滑块移动游标", interactive=True)
                with gr.Row():
                    arm_plot = gr.BarPlot(label="双臂姿态角", x="帧号", y="双臂姿态角", color='动作环节')
                    spine_plot = gr.LinePlot(
                        label="脊柱倾角", 
                        x="帧号", 
                        y="脊柱倾角",
                        color="警告",
                        color_map={
                            "角度": "#2196f3",
                            "+5°警戒线": "#ff0000",
                            "-5°警戒线": "#ff0000",
                        },
                        overlay_point=True,
                    )
                with gr.Row():
                    angular_velocity_plot = gr.LinePlot(label="双臂角速度", x="帧号", y="角速度")
                    angular_acceleration_plot = gr.LinePlot(label="双臂角加速度", x="帧号", y="角加速度")
                with gr.Row():
                    phase_plot = gr.ScatterPlot(label="相位图", x="双臂姿态角", y="角速度")
            
        process_btn.click(
            fn=lambda user_options, x: user_options.update({'device_dropdown': x}), inputs=[user_options, device_dropdown], outputs=[user_options]
        ).then(
            fn=lambda user_options, x: user_options.update({'bow_hand': x}), inputs=[user_options, bow_hand], outputs=[user_options]
        ).then(
            fn=lambda user_options, x: user_options.update({'model_dropdown': x}), inputs=[user_options, model_dropdown], outputs=[user_options]
        ).then(
            fn=lambda user_options, x: user_options.update({'batch_size': x}), inputs=[user_options, batch_size], outputs=[user_options]
        ).then(
            fn=process_video,
            inputs=[input_video, user_options],
            outputs=[status_text, output_video, slider, current_frame, arm_plot, spine_plot, angular_velocity_plot, angular_acceleration_plot, phase_plot]
        )
        
        slider.change(
            fn=Video.extract_frame,inputs=[output_video, slider],outputs=[current_frame]
        ).then(
            fn=lambda df, idx: list(zip(df['columns'], df['data'][idx])),
            inputs=[arm_plot, slider],
            outputs=[current_frame_data]
        )

        refresh_btn.click(
            fn=lambda df, idx: list(zip(df['columns'], df['data'][idx])),
            inputs=[arm_plot, slider],
            outputs=[current_frame_data]
        ).then(
            fn=update_cursor, inputs=[arm_plot, slider], outputs=[arm_plot]
        ).then(
            fn=update_cursor, inputs=[spine_plot, slider], outputs=[spine_plot]
        ).then(
            fn=update_cursor, inputs=[angular_velocity_plot, slider], outputs=[angular_velocity_plot]
        ).then(
            fn=update_cursor, inputs=[angular_acceleration_plot, slider], outputs=[angular_acceleration_plot]
        ).then(
            fn=update_cursor, inputs=[phase_plot, slider], outputs=[phase_plot]
        )

    # todo 头部姿态角 头部与脊柱的夹角：
    # 关键错误姿势示例
    # 前肩耸肩（肩角 < 150°）：导致肩部疲劳，箭着点偏低。
    # 后肘塌陷（肘角 < 100°）：力量分散，撒放不干脆。
    # 脊柱侧倾（左右倾斜 > 5°）：影响瞄准一致性，长期导致腰背劳损。
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
