import nltk
import os
import shutil
import gradio as gr

from service.config import *
from service.langchain_application import LangChainApplication

nltk.data.path = [NLTK_DATA_PATH] + nltk.data.path

# 知识库名称和路径
knowledge_map = {
    "本地知识库": DATA_VECTOR_STORE_PATH + "init_local"
}
# 启动
application = LangChainApplication()
# 加载知识库
application.source_service.load_vector_store(knowledge_map["本地知识库"])


# 上传文件
def upload_file(file):
    if not os.path.exists("docs"):
        os.mkdir("docs")
    filename = os.path.basename(file.name)
    shutil.move(file.name, "docs/" + filename)
    application.source_service.add_document("docs/" + filename)


# 本地知识库文档列表
def local_file_list():
    file_list = []
    for doc in os.listdir(DATA_UPLOAD_PATH):
        filepath = DATA_UPLOAD_PATH + doc
        if filepath.endswith(".doc") or filepath.endswith(".docx"):
            file_list.append(filepath)
    return file_list
v_local_file_list = local_file_list()


# 切换/加载 知识库
def set_knowledge(kg_name, history):
    try:
        application.source_service.load_vector_store(knowledge_map[kg_name])
        msg_status = f'{kg_name}知识库已成功加载'
    except Exception as e:
        print(e)
        msg_status = f'{kg_name}知识库未成功加载'
    return history + [[None, msg_status]]


def clear_session():
    return '', [], []


def predict(input,
            top_k,
            doc_score,
            use_pattern,
            chatbot,
            history,
            max_token,
            top_p,
            temperature):
    if history == None:
        history = []
    if chatbot == None:
        chatbot = []
    # if use_web == '使用':
    #     web_content = application.source_service.search_web(query=input)
    # else:
    #     web_content = ''
    chatbot.append((input, ""))
    if use_pattern == '模型问答':
        for resp, history in application.get_llm_answer(input, history, "", max_token, temperature, top_p):
            chatbot[-1] = (input, resp)
            yield '', chatbot, history
    else:
        if application.source_service.vector_store:
            for resp, history in application.get_knowledge_based_answer(input, history, "", max_token, temperature, top_p, doc_score, top_k):
                chatbot[-1] = (input, resp["result"])
                yield '', chatbot, history
            source = resp["result"] + "\n\n"
            source += "".join(
                [
                    f"""<details> <summary>出处 [{i + 1}] {os.path.split(doc.metadata["source"])[-1]} 相似度 {doc.metadata["score"]}</summary>\n"""
                    f"""{doc.page_content}\n"""
                    f"""</details>"""
                    for i, doc in
                    enumerate(resp["source_documents"])])
            chatbot[-1] = (input, source)
            yield '', chatbot, history
        else:
            chatbot[-1] = (input, "请先加载知识库！！")
            yield '', chatbot, history


# with open("assets/custom.css", "r", encoding="utf-8") as f:
#     customCSS = f.read()
with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>LangChain-ChatGLM2-6B</center></h1>
        <center><font size=3>
        </center></font>
        """)
    history = gr.State([])

    with gr.Row():
        # 左栏：配置
        with gr.Column(scale=1):
            use_pattern = gr.Radio(
                ['模型问答', '知识库问答'],
                label="模式",
                value='模型问答',
                interactive=True)

            top_k = gr.Slider(1, 5, value=1, step=1, label="检索top-k文档", interactive=True)
            doc_score = gr.Slider(300, 1000, value=800, step=1, label="文档匹配分数", interactive=True)

            kg_name = gr.Dropdown(
                list(knowledge_map.keys()),
                label="知识库",
                info="使用知识库问答，请加载知识库",
                value="本地知识库",
                interactive=False)

            gr.Dropdown(
                list(v_local_file_list),
                label="已加载文档",
                info="加载更多文档，请联系管理员",
                value=v_local_file_list[0],
                interactive=True)

            file = gr.File(
                label="将文件上传到知识库库，内容要尽量匹配",
                visible=True,
                file_types=['.txt', '.md', '.docx', '.pdf'],
                interactive=False)

        # 中间栏：对话
        with gr.Column(scale=4):
            chatbot = gr.Chatbot().style(height=600)
            with gr.Row():
                with gr.Column(scale=4):
                    message = gr.Textbox(label='请输入问题', lines=4)
                    with gr.Row():
                        clear_history = gr.Button("🧹 清除历史对话")
                        send = gr.Button("🚀 发送")
                with gr.Column(scale=1):
                    max_token = gr.Slider(100, 10000, value=8192, step=1, label="Maximum length", interactive=True)
                    top_p = gr.Slider(0.1, 1, value=0.8, step=0.01, label="Top p", interactive=True)
                    temperature = gr.Slider(0.1, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

        # ============= 触发动作=============
        # 上传文件
        file.upload(upload_file,
                    inputs=file,
                    outputs=None)

        # 切换知识库
        kg_name.change(
            set_knowledge,
            show_progress=True,
            inputs=[kg_name, chatbot],
            outputs=chatbot)

        # 提交输入
        send.click(
            predict,
            inputs=[
                message,
                top_k,
                doc_score,
                use_pattern,
                chatbot,
                history,
                max_token,
                top_p,
                temperature
            ],
            outputs=[message, chatbot, history])

        # 清空历史对话按钮 提交
        clear_history.click(
            fn=clear_session,
            inputs=[],
            outputs=[message, chatbot, history],
            queue=False)

demo.queue(concurrency_count=10).launch(
    server_name='0.0.0.0',
    server_port=7888,
    share=False,
)
