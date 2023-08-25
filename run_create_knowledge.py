import os
import pandas as pd
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from tqdm import tqdm
from service.config import *
from service.service_verctor import VectorService
from langchain.schema import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from service.tools.text_splitter import ChineseTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from service.config import *
from langchain.document_loaders import PyPDFLoader
from pdf2docx import Converter
import nltk

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH, model_kwargs={'device': EMBEDDING_DEVICE})

# ============================示例：创建本地知识库====================================================
# docs = []
# for doc in os.listdir(DATA_UPLOAD_PATH):
#     if doc.endswith('.txt'):
#         print(doc)
#         loader = UnstructuredFileLoader(f'{DATA_UPLOAD_PATH}/{doc}', mode="elements")
#         doc = loader.load()
#         docs.extend(doc)
# vector_store = FAISS.from_documents(docs, embeddings)
# vector_store.save_local(DATA_VECTOR_STORE_PATH + "init_local")


# ============================示例：创建本地知识库，自定义文档加载与分割====================================================
docs = []
for doc in os.listdir(DATA_UPLOAD_PATH):
    filepath = DATA_UPLOAD_PATH + doc
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,
        chunk_overlap=0,
        length_function=len,
    )

    if doc.endswith('.txt'):
        tools = UnstructuredFileLoader(filepath, mode="elements")
        doc = tools.load()
        # for x in doc:
        #     print(x)
        docs.extend(doc)
    #
    # if filepath.lower().endswith(".md"):
    #     tools = UnstructuredMarkdownLoader(filepath)
    #     doc = tools.load_and_split(text_splitter)
    #     for x in doc:
    #         print(x)
    #     docs.extend(doc)

    # if filepath.lower().endswith(".pdf"):
    #     docx = filepath[0:-3] + "docx"
    #     if not os.path.exists(docx):
    #         cv = Converter(filepath)
    #         cv.convert(docx)
    #         cv.close()

    if filepath.endswith(".doc") or filepath.endswith(".docx"):
        loader = UnstructuredFileLoader(filepath)
        text_splitter = ChineseTextSplitter(reg_list=["\n\n(?=\d+)", "\n\n", "\n", " "], max_len=500)
        doc = loader.load_and_split(text_splitter)
        for x in doc:
            print(x)
        docs.extend(doc)

vector_store = FAISS.from_documents(docs, embeddings)
vector_store.save_local(DATA_VECTOR_STORE_PATH + "init_local")

# ============================示例：其他====================================================
# Wikipedia数据处理

# docs = []

# with open('docs/zh_wikipedia/zhwiki.sim.utf8', 'r', encoding='utf-8') as f:
#     for idx, line in tqdm(enumerate(f.readlines())):
#         metadata = {"source": f'doc_id_{idx}'}
#         docs.append(Document(page_content=line.strip(), metadata=metadata))
#
# vector_store = FAISS.from_documents(docs, embeddings)
# vector_store.save_local('vector_store/zh_wikipedia/')


# docs = []
#
# with open('vector_store/zh_wikipedia/wiki.zh-sim-cleaned.txt', 'r', encoding='utf-8') as f:
#     for idx, line in tqdm(enumerate(f.readlines())):
#         metadata = {"source": f'doc_id_{idx}'}
#         docs.append(Document(page_content=line.strip(), metadata=metadata))
#
# vector_store = FAISS.from_documents(docs, embeddings)
# vector_store.save_local('vector_store/zh_wikipedia/')


# # 金融研报数据处理
# docs = []
#
# for doc in tqdm(os.listdir(docs_path)):
#     if doc.endswith('.txt'):
#         # print(doc)
#         # tools = UnstructuredFileLoader(f'{docs_path}/{doc}', mode="elements")
#         # doc = tools.load()
#         f=open(f'{docs_path}/{doc}','r',encoding='utf-8')
#
#         # docs.extend(doc)
#         docs.append(Document(page_content=''.join(f.read().split()), metadata={"source": f'doc_id_{doc}'}))
# vector_store = FAISS.from_documents(docs, embeddings)
# vector_store.save_local('vector_store/financial_research_reports')


# # 英雄联盟
#
# docs = []
#
# lol_df = pd.read_csv('vector_store/lol/champions.csv')
# # lol_df.columns = ['id', '英雄简称', '英雄全称', '出生地', '人物属性', '英雄类别', '英雄故事']
# print(lol_df)
#
# for idx, row in lol_df.iterrows():
#     metadata = {"source": f'doc_id_{idx}'}
#     text = ' '.join(row.values)
#     # for col in ['英雄简称', '英雄全称', '出生地', '人物属性', '英雄类别', '英雄故事']:
#     #     text += row[col]
#     docs.append(Document(page_content=text, metadata=metadata))
#
# vector_store = FAISS.from_documents(docs, embeddings)
# vector_store.save_local('vector_store/lol/')
