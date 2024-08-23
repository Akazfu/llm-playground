from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI()
# print(model.model_name)
prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
chain = prompt | model

# The input schema of the chain is the input schema of its first part, the prompt.
# print(chain.input_schema.schema())
# print(prompt.input_schema.schema())
# print(model.input_schema.schema())
# print(prompt.output_schema.schema())


# for s in chain.stream({"topic": "bears"}):
#     print(s.content, end="", flush=True)

print(chain.invoke({"topic": "bears"}))