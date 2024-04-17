from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os

load_dotenv()

# Load Acces token From .env
HUGGING_FACE_ACCESS_TOKEN = os.getenv('HUGGING_FACE_ACCESS_TOKEN')
print(HUGGING_FACE_ACCESS_TOKEN)

repo_id="mistralai/Mistral-7B-Instruct-v0.2"

template="""
Question: {question}
Answer: Let's think step by step
"""
question= "Can you give me a diet cold coffee recipe that I can make within 5 minutes?"

prompt = PromptTemplate.from_template(template=template)
print(prompt)

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=HUGGING_FACE_ACCESS_TOKEN
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

response = llm_chain.invoke(question)

print(response)