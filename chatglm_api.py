from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer
from typing import List
import uvicorn


app = FastAPI()
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b",
                                          trust_remote_code=True
                                          )
model = AutoModel.from_pretrained("THUDM/chatglm-6b",
                                  trust_remote_code=True
                                  ).half().cuda()
model = model.eval()


@app.get("/")
def hello():
    return {"message": "Hello ChatGLM API!"}


@app.post("/predict")
def pred_chat(user_msg: str,
              history: List[List[str]]):
    response, history = model.chat(tokenizer, user_msg, history)
    return {"response": response,
            "history": history}


if __name__ == "__main__":
    uvicorn.run(app="chatglm_api:app",
                host="127.0.0.1",
                port=8080,
                reload=True)