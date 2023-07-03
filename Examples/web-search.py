from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv()) #automatically find .env file in directory.

from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.tools import Tool, DuckDuckGoSearchRun
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)

web_search = DuckDuckGoSearchRun()

tools = [
    Tool(
        name = "Search",
        func=web_search.run,
        description="useful for when you search query on web"
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

output = agent.run("Who is the president of india?")

print("Result: ", output)