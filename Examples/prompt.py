from langchain import OpenAI

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

llm = OpenAI(temperature=0) #temperature decides the randomness or creativity of model, ranges from 0 to 1.

name = llm("I want to open a restaurant for indian food. suggest a great name for this.")

print(name)