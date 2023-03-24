from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer
from typing import List
import uvicorn
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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
              history: List[List[str]],
              max_length: int = 2048,
              num_beams: int = 1,
              do_sample: bool = True,
              top_p: float = 0.7,
              temperature: float = 0.95,
              ):
    response, history = model.chat(tokenizer,
                                   user_msg,
                                   history,
                                   max_length=max_length,
                                   top_p=top_p,
                                   temperature=temperature
                                   )
    #clear gpu cache
    torch_gc()
    return {"response": response,
            "history": history}


if __name__ == "__main__":
    uvicorn.run(app=app,
                host="127.0.0.1",
                port=8080,
                reload=True)
