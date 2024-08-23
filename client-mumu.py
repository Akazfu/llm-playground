from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Load .env
load_dotenv()

prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
model = ChatOpenAI(model="mumu-med-v0.1", temperature=0)

output_parser = StrOutputParser()

chain = prompt | model | output_parser

# inputs_batch = [{"topic": topic} for topic in ["ice cream", "spaghetti", "dumplings"]]
# responses = chain.batch(inputs_batch)
# for response in responses:
#     print(response)


response = chain.stream({"topic": "ice cream"})
for chunk in response:
    print(chunk, end="", flush=True)


# prompt_value = prompt.invoke({"topic": "ice cream"})


# ChatPromptValue(messages=[HumanMessage(content="tell me a short joke about ice cream")])

# prompt_value.to_messages()
