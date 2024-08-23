import os

import openai
from dotenv import find_dotenv, load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

_ = load_dotenv(find_dotenv())



# def get_completion(prompt, model="gpt-3.5-turbo"):
#     messages = [{"role": "user", "content": prompt}]
#     response = openai.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.7,
#     )
#     return response.choices[0].message.content


# response = get_completion("你现在是什么版本？gpt4还是gpt3.5？")
# print(response)

# give me a chinese email string

customer_email = "你好，我是你的客户。我想问一下你们的产品怎么样？ 你们的产品好不好？"
style = "Japanese in a calm and respectful tone"

# prompt = f"Translate the following Chinese email into {style}: “”“{customer_email}”“”"

# print(prompt)
# print(get_completion(prompt))

# LangChain
chat = ChatOpenAI(temperature=0.7)
template_string = (
    "Translate the following Chinese email into {style}: “”“{customer_email}”“”"
)
prompt_template = ChatPromptTemplate.from_template(template_string)

customer_messages = prompt_template.format_messages(
    style=style, customer_email=customer_email
)

customer_response = chat.invoke(customer_messages)
print(customer_response.content)
