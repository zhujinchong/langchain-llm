import os
from duckduckgo_search import ddg
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from service.config import *


class VectorService(object):
    def __init__(self):
        self.vector_store = None
        self.docs_path = DATA_UPLOAD_PATH
        self.vector_store_path = DATA_VECTOR_STORE_PATH
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={'device': EMBEDDING_DEVICE})

    # 创建本地知识库向量
    def init_source_vector(self):
        docs = []
        for doc in os.listdir(self.docs_path):
            if doc.endswith('.txt'):
                # print(doc)
                loader = UnstructuredFileLoader(f'{self.docs_path}/{doc}', mode="elements")
                doc = loader.load()
                docs.extend(doc)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        self.vector_store.save_local(self.vector_store_path)

    # 切换知识库
    def load_vector_store(self, path):
        if path is None:
            self.vector_store = FAISS.load_local(self.vector_store_path, self.embeddings)
        else:
            self.vector_store = FAISS.load_local(path, self.embeddings)
        return self.vector_store

    # 知识库中添加文档
    def add_document(self, document_path):
        loader = UnstructuredFileLoader(document_path, mode="elements")
        doc = loader.load()
        self.vector_store.add_documents(doc)
        self.vector_store.save_local(self.vector_store_path)

    def search_web(self, query):
        # SESSION.proxies = {
        #     "http": f"socks5h://localhost:7890",
        #     "https": f"socks5h://localhost:7890"
        # }
        try:
            results = ddg(query)
            web_content = ''
            if results:
                for result in results:
                    web_content += result['body']
            return web_content
        except Exception as e:
            print(f"网络检索异常:{query}")
            return ''


# if __name__ == '__main__':
#     config = LangChainCFG()
#     source_service = VectorService(config)
#     source_service.init_source_vector()
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
#
#     source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/added/科比.txt')
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
#
#     vector_store=source_service.load_vector_store()
#     search_result = source_service.vector_store.similarity_search_with_score('科比')
#     print(search_result)
