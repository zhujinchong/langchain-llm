import os


EMBEDDING_MODEL_PATH = "/data/zjc/embeddings/text2vec-large-chinese"
EMBEDDING_DEVICE_ID = "0"
EMBEDDING_DEVICE = "cuda:" + EMBEDDING_DEVICE_ID


LLM_MODEL_PATH = "/data/zjc/chatglm2/chatglm2-6b-f16"
# LLM_DEVICE_ID = "0"
# LLM_DEVICE = "cuda:" + LLM_DEVICE_ID
LLM_MAX_HISTORY_LEN = 5


DATA_UPLOAD_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/upload_docs", "")
DATA_VECTOR_STORE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/vector_store", "")


PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知内容:
{context}
问题:
{question}"""

PROMPT_WEB_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知网络检索内容：{web_content}
已知内容:
{context}
问题:
{question}"""

NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "third_package/nltk_data")