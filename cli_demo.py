import platform
import requests

os_name = platform.system()

history = []
print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
while True:
    query = input("\n用户：")
    if query == "stop":
        break
        history = []
        command = 'cls' if os_name == 'Windows' else 'clear'
        os.system(command)
        print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
        continue
    resp = requests.post(f"http://127.0.0.1:8080/predict?user_msg={query}", json=history)
    if resp.status_code == 200:
        response, history = resp.json()["response"], resp.json()["history"]
    else:
        response, history = "请求异常，请稍后再试", history
    print(f"ChatGLM-6B：{response}")
