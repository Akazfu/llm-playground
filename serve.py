#!/usr/bin/env python
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve import RemoteRunnable, add_routes
from pydantic import BaseModel as PydanticBaseModel

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools.retriever import create_retriever_tool

# Load .env
load_dotenv()

# 1. Load Retriever
loader = WebBaseLoader(
    [
        "https://movie.douban.com/subject/36151692/?from=showing",
        "https://movie.douban.com/subject/36081094/?from=showing",
        "https://movie.douban.com/subject/35575567/?from=showing",
    ]
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# 2. Create Tools
retriever_tool = create_retriever_tool(
    retriever,
    "Douban_search",
    "Search for information about Douban. For any questions about Douban, you must use this tool!",
)
search = TavilySearchResults()
tools = [retriever_tool, search]

# 3. Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 4. App definition
app = FastAPI(
    title="Douban Server",
    version="1.0",
    description="A simple API server using Douban's Runnable interfaces",
)

# 5. Adding chain route

# We need to add these input/output schemas because the current AgentExecutor
# is lacking in schemas.


class PromptRequest(PydanticBaseModel):
    input: str
    chat_history: List[str]


class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )


class Output(BaseModel):
    output: str


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)


@app.post("/promptrequest")
def submit_request(request: PromptRequest):
    remote_chain = RemoteRunnable("http://localhost:8000/agent/")
    print(request.input)
    print(request.chat_history)
    result = remote_chain.invoke(
        {
            "input": request.input,
            "chat_history": request.chat_history,
        }
    )
    return {"output": result["output"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
