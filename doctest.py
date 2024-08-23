import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load .env
load_dotenv()

# LLM invocation
llm = ChatOpenAI()
# response = llm.invoke("我是你爸，千变万化。我是你爹，法力无边！")
# print(response.content)

# Prompt Template Chain
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "你是一个非常叛逆的女儿"),
#         ("user", "{input}"),
#     ]
# )
# chain = prompt | llm
# response = chain.invoke({"input": "我是你爸，千变万化。我是你爹，法力无边！"})
# print(response.content)

# Output Parser Chain
# output_parser = StrOutputParser()
# chain = prompt | llm | output_parser
# response = chain.invoke({"input": "我是你爸，千变万化。我是你爹，法力无边！"})
# print(response)

# Retrieval Chain
loader = WebBaseLoader(
    [
        "https://movie.douban.com/subject/36151692/?from=showing",
        "https://movie.douban.com/subject/36081094/?from=showing",
        "https://movie.douban.com/subject/35575567/?from=showing",
    ]
)
docs = loader.load()
# Embeddings
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context in chinese:

<context>
{context}
</context>

Question: {input}"""
)

# document_chain = create_stuff_documents_chain(llm, prompt)

# 手动导入例子
# response = document_chain.invoke(
#     {
#         "input": "阿花是谁?",
#         "context": [Document(page_content="阿花是一只小狗")],
#     }
# )

# print(response)

retriever = vector.as_retriever()
# retrieval_chain = create_retrieval_chain(retriever, document_chain)

# response = retrieval_chain.invoke({"input": "周三除三害这部电影讲了什么?有哪些演员"})
# print(response["answer"])


prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        (
            "user",
            "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
        ),
    ]
)

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

from langchain_core.messages import AIMessage, HumanMessage

chat_history = [
    HumanMessage(content="最近上映了什么电影？"),
    AIMessage(content="周处除三害，沙丘2，热辣滚烫"),
]
retriever_chain.invoke(
    {"chat_history": chat_history, "input": "这个电影是什么时候上映，豆瓣评分怎么样？"}
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's questions based on the below context:\n\n{context} in chinese. \
            注意豆瓣评分要着重参考context里类似这段字符串'豆瓣评分\n            \n\n\n引用\n\n\n\n分数'",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)
document_chain = create_stuff_documents_chain(llm, prompt)

retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# response = retrieval_chain.invoke(
#     {
#         "chat_history": chat_history,
#         "input": "这几部电影是什么时候上映，豆瓣评分怎么样？每一部电影给一个精选短评。",
#     }
# )
# print(response["answer"])

# AGENT
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "Douban_search",
    "Search for information about Douban. For any questions about Douban, you must use this tool!",
)

from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults()
tools = [retriever_tool, search]
# result = search.invoke({"query": "现在是北京时间几点"})
# print(result)

from langchain_openai import ChatOpenAI

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# result = agent_executor.invoke({"input": "ipad pro新版什么时候发布？"})
# print(result["output"])

chat_history = [
    HumanMessage(content="最近上映了什么电影？"),
    AIMessage(content="《热辣滚烫》"),
]
result = agent_executor.invoke(
    {
        "chat_history": chat_history,
        "input": "这部电影是什么时候上映，豆瓣评分怎么样？每一部电影给一个精选短评。",
    }
)
print(result["output"])
