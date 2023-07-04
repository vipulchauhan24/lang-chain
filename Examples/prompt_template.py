from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# temperature decides the randomness or creativity of model, ranges from 0 to 1.
llm = OpenAI(temperature=0)

template = """
I want to open a restaurant for {cuisine} food. suggest a great name for this.
"""

prompt = PromptTemplate(input_variables=["cuisine"], template=template)
# prompt.format(cuisine="indian")

name = LLMChain(prompt=prompt, llm=llm)

print(name.run("italian"))
