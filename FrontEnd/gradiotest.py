import gradio as gr
from control import move_robot
from Whispertest import transcribe_keywords
from command_parser import execute_motion
from llm_api import voice_to_llm, text_to_llm


with gr.Blocks(theme=gr.themes.Default(), css=".gradio-container {background: #f7f9fa;} .gr-button {margin: 2px 2px;} .gr-textbox {margin-bottom: 6px;} .gr-column {min-width: 200px; max-width: 520px;} .gr-row {flex-wrap: wrap;} .gradio-container {overflow-y: auto; max-height: 98vh;}") as demo:
    gr.Markdown("""
    <h2 style='text-align:center; color:#2d3a4b; margin-bottom:8px;'>GUI</h2>
    <hr style='margin-bottom: 8px;'>
    """)
    with gr.Row():
        # 左栏：机械臂控制、语音识别、状态
        with gr.Column(scale=1, min_width=180, elem_id="left-col"):
            gr.Markdown("<h4 style='color:#3b5998;'>机械臂方向/旋转控制</h4>")
            with gr.Row():
                up_btn = gr.Button("↑", elem_id="up-btn")
            with gr.Row():
                left_btn = gr.Button("←", elem_id="left-btn")
                right_btn = gr.Button("→", elem_id="right-btn")
            with gr.Row():
                down_btn = gr.Button("↓", elem_id="down-btn")
            with gr.Row():
                rplus_btn = gr.Button("R+", elem_id="rplus-btn")
                rminus_btn = gr.Button("R-", elem_id="rminus-btn")
            gr.Markdown("<hr style='margin:8px 0;'>")
            gr.Markdown("<h4 style='color:#3b5998;'>语音转文字（Whisper）</h4>")
            audio_in = gr.Audio(type="filepath", label="请录音或上传音频", elem_id="audio-in")
            with gr.Row():
                recog_btn = gr.Button("识别音频", elem_id="recog-btn")
                exec_btn = gr.Button("执行指令", elem_id="exec-btn")
            recog_out = gr.Textbox(label="筛选后指令", interactive=True, elem_id="recog-out")
            exec_out = gr.Textbox(label="执行结果", elem_id="exec-out")

        # 中栏：视频流
        with gr.Column(scale=1, min_width=240, elem_id="center-col"):
            gr.Markdown("<h4 style='color:#3b5998;'>实时画面</h4>")
            gr.HTML('<div style="display:flex; justify-content:center;"><img src="http://localhost:5001/video_feed_thumb" width="480" height="360" style="border-radius:8px; box-shadow:0 2px 8px #ccc;" /></div>')
            gr.Markdown("<hr style='margin:8px 0;'>")
            gr.Markdown("<h4 style='color:#3b5998;'>大模型对话</h4>")
            chatbot = gr.Chatbot(label="对话历史", type="messages", elem_id="chatbot", height=220)
            gr.Markdown("<hr style='margin:8px 0;'>")
            status_box = gr.Textbox(label="机械臂状态", value="等待操作...", interactive=False, elem_id="status-box")
        # 右栏：大模型对话
        with gr.Column(scale=1, min_width=200, elem_id="right-col"):

            gr.Markdown("<hr style='margin:8px 0;'>")
            gr.Markdown("<b>语音输入</b>")
            with gr.Row():
                audio_chat = gr.Audio(type="filepath", label="语音输入", elem_id="audio-chat")
                talk_btn = gr.Button("语音识别", elem_id="talk-btn")
            gr.Markdown("<b>输入</b>")
            with gr.Row():
                text_input = gr.Textbox(label="输入", elem_id="text-input")
                send_btn = gr.Button("发送", elem_id="send-btn")
            chat_status = gr.Textbox(label="对话状态", interactive=False, elem_id="chat-status")
            gr.Markdown("<hr style='margin:8px 0;'>")
    # 按钮事件绑定
    up_btn.click(lambda: move_robot('up'), outputs=status_box)
    down_btn.click(lambda: move_robot('down'), outputs=status_box)
    left_btn.click(lambda: move_robot('left'), outputs=status_box)
    right_btn.click(lambda: move_robot('right'), outputs=status_box)
    rplus_btn.click(lambda: move_robot('r+'), outputs=status_box)
    rminus_btn.click(lambda: move_robot('r-'), outputs=status_box)
    recog_btn.click(transcribe_keywords, inputs=audio_in, outputs=recog_out)
    exec_btn.click(lambda filtered: execute_motion(filtered), inputs=recog_out, outputs=exec_out)
    talk_btn.click(voice_to_llm, inputs=[audio_chat, chatbot], outputs=[chatbot, chat_status])
    send_btn.click(text_to_llm, inputs=[text_input, chatbot], outputs=[chatbot, chat_status])

demo.launch()
