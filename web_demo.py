import requests
import gradio as gr


MAX_TURNS = 20
MAX_BOXES = MAX_TURNS * 2


def predict(input, history=None):
    if history is None:
        history = []
    resp = requests.post(f"http://127.0.0.1:8080/predict?user_msg={input}", json=history)
    if resp.status_code == 200:
        response, history = resp.json()["response"], resp.json()["history"]
    else:
        response, history = "请求异常，请稍后再试", history
    updates = []
    for query, response in history:
        updates.append(gr.update(visible=True, value="用户：" + query))
        updates.append(gr.update(visible=True, value="ChatGLM-6B：" + response))
    if len(updates) < MAX_BOXES:
        updates = updates + [gr.Textbox.update(visible=False)] * (MAX_BOXES - len(updates))
    return [history] + updates


with gr.Blocks() as demo:
    state = gr.State([])
    text_boxes = []
    for i in range(MAX_BOXES):
        if i % 2 == 0:
            text_boxes.append(gr.Markdown(visible=False, label="提问："))
        else:
            text_boxes.append(gr.Markdown(visible=False, label="回复："))

    with gr.Row():
        with gr.Column(scale=4):
            txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
        with gr.Column(scale=1):
            button = gr.Button("Generate")
    button.click(predict, [txt, state], [state] + text_boxes)
demo.queue().launch(share=False,
                    server_name="0.0.0.0")
