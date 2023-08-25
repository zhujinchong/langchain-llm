## 介绍

基于LangChain和ChatGLM2-6B实现的本地知识库

## 🔥 效果演示

![db0615cef192263bd1343e7b9e0032cc.png](images%2Fdb0615cef192263bd1343e7b9e0032cc.png)

## 🏗️ 开发部署

### 1 硬件需求

LLM模型：ChatGLM2-6B-int4需要7G显存；ChatGLM2-6B需要13G

Embedding模型：GanymedeNil/text2vec-large-chinese需要2G显存

### 2 下载模型至本地

从Github/Huggingface下载模型至本地（方法略）

### 3 安装环境

安装基本环境

```shell
pip install -r requirements.txt
```

此外，torch需要再次安装GPU版本！

个别依赖，如pdf2docx，使用过程遇到环境问题自行解决。

### 4 修改配置

修改config.py中的配置：

* Embedding模型位置
* LLM模型位置

```python
EMBEDDING_MODEL_PATH = "/data/embeddings/text2vec-large-chinese"
LLM_MODEL_PATH = "/data/chatglm2/chatglm2-6b-f16"
```

### 5 运行

创建本地知识库

```shell
python run_create_knowledge.py
```

运行webui

```shell
python run_app.py
```


## 🔨变更日志

### v0.1版本

1. 支持非结构化文档（已支持 md、pdf、docx、txt 文件格式）
2. 支持搜索引擎 
3. LLM模型支持ChatGLM-6B系列 
4. Embedding模型支持GanymedeNil/text2vec-large-chinese 
5. 基于 gradio 实现 Web UI DEMO


## ❤️引用

langchain参考：https://github.com/yanqiangmiffy/Chinese-LangChain

langchain参考：https://github.com/yanqiangmiffy/Chinese-LangChain

LLM模型：https://github.com/THUDM/ChatGLM-6B
