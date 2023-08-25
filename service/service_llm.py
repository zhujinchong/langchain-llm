import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from typing import List

from langchain.llms.base import LLM
from transformers import AutoModel, AutoTokenizer
from service.config import *
from service.tools.utils import torch_gc


class ChatGLMService(LLM):
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def load_model(self):
        model_name_or_path = LLM_MODEL_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
        self.model = self.model.eval()

    # def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
    #     response, _ = self.model.chat(self.tokenizer, prompt)
    #     if stop is not None:
    #         response = enforce_stop_tokens(response, stop)
    #     return response

    # 流式回答
    def _call(self, prompt: str, history: List[List[str]] = []) -> str:
        # history: List[List[str]] = [], max_token = 2048, top_p = 0.8, temperature = 0.95
        max_token = 2048
        top_p = 0.8
        temperature = 0.95
        _history = history[-LLM_MAX_HISTORY_LEN:]
        _history = [tuple(h) for h in _history]
        for inum, (resp, _) in enumerate(self.model.stream_chat(
                self.tokenizer, prompt, history=_history, max_length=max_token, top_p=top_p, temperature=temperature)):
            if inum == 0:
                history += [[prompt, resp]]
            else:
                history[-1] = [prompt, resp]
            yield resp, history
        torch_gc()


# if __name__ == '__main__':
#     llm = ChatGLMService()
#     llm.load_model()
#     history = []
#     while True:
#         query = input("请输入您的问题：")
#         last_print_len = 0
#         for resp, _history in llm._call(query, history):
#             print(resp[last_print_len:], end="\n", flush=True)
#             last_print_len = len(resp)
#             history = _history
