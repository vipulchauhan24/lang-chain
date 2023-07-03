from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel, Field, validator
from langchain.callbacks.stdout import StdOutCallbackHandler
from langchain.chains import create_extraction_chain_pydantic
from typing import Optional, List
from pydantic import BaseModel, Field

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv()) #automatically find .env file in directory.

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")


class Properties(BaseModel):
    person_name: str
    person_height: int
    person_hair_color: str
    dog_breed: Optional[str]
    dog_name: Optional[str]

inp = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
"""

chain = create_extraction_chain_pydantic(pydantic_schema=Properties, llm=llm)    

output=chain.run(inp, callbacks=[StdOutCallbackHandler()])

print(output)

