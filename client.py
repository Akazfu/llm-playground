from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/agent/")
result = remote_chain.invoke(
    {
        "input": "《热辣滚烫》这部电影是什么时候上映，豆瓣评分怎么样？每一部电影给一个精选短评。",
        "chat_history": [],  # Providing an empty list as this is the first call
    }
)

print(result["output"])
