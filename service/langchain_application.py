from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

from service.service_llm import ChatGLMService
from service.service_verctor import VectorService
from service.config import *
from typing import List


def generate_prompt(query: str, related_docs: List[str], web_content) -> str:
    prompt = ''
    if web_content:
        context = "\n".join([doc.page_content for doc in related_docs])
        PROMPT_WEB_TEMPLATE.replace("{question}", query).replace("{context}", context).replace("{web_content}",
                                                                                               web_content)
    else:
        context = "\n".join([doc.page_content for doc in related_docs])
        prompt = PROMPT_TEMPLATE.replace("{question}", query).replace("{context}", context)
    return prompt


def get_docs_with_score(docs_with_score, doc_score):
    docs = []
    for doc, score in docs_with_score:
        doc.metadata["score"] = score
        if score <= doc_score:
            docs.append(doc)
    return docs


class LangChainApplication(object):
    def __init__(self):
        self.llm_service = ChatGLMService()
        self.llm_service.load_model()
        self.source_service = VectorService()

    def get_knowledge_based_answer(
            self,
            query,
            history=[],
            web_content='',
            max_token=10000,
            temperature=0.95,
            top_p=0.8,
            doc_score=800,
            top_k=4):

        related_docs_with_score = self.source_service.vector_store.similarity_search_with_score(query, k=top_k)
        related_docs = get_docs_with_score(related_docs_with_score, doc_score)
        prompt = generate_prompt(query, related_docs, web_content)
        # print("----------")
        # for chat in chat_history:
        # print(chat)
        for result, history in self.llm_service._call(prompt, history):
            history[-1][0] = query
            response = {"query": query,
                        "result": result,
                        "source_documents": related_docs}
            yield response, history

    def get_llm_answer(self,
                       query='',
                       history=[],
                       web_content='',
                       max_token=10000,
                       temperature=0.95,
                       top_p=0.8):
        if web_content:
            prompt = '基于网络检索内容：' + web_content + '\n回答以下问题：' + query
        else:
            prompt = query
        for resp, history in self.llm_service._call(prompt, history):
            yield resp, history

# if __name__ == '__main__':
#     application = LangChainApplication()
#     # result = application.get_knowledge_based_answer('马保国是谁')
#     # print(result)
#     # application.source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/added/马保国.txt')
#     # result = application.get_knowledge_based_answer('马保国是谁')
#     # print(result)
#     result = application.get_llm_answer('马保国是谁')
#     print(result)
