from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.callbacks.stdout import StdOutCallbackHandler

from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv()) #automatically find .env file in directory.

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

schema = {
    "properties": {
        "person_name": {"type": "string"},
        "person_height": {"type": "integer"},
        "person_hair_color": {"type": "string"},
        "dog_name": {"type": "string"},
        "dog_breed": {"type": "string"},
    },
    "required": ["person_name", "person_height"],
}

inp = """
Alex is 5 feet tall. Claudia is 1 feet taller Alex and jumps higher than him. Claudia is a brunette and Alex is blonde.
Alex's dog Frosty is a labrador and likes to play hide and seek.
"""

chain = create_extraction_chain(schema=schema, llm=llm)

output=chain.run(inp, callbacks=[StdOutCallbackHandler()])

print(output)