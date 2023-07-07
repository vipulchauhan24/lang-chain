from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# temperature decides the randomness or creativity of model, ranges from 0 to 1.
llm = OpenAI(temperature=0.5)


def generate_restaurant_name_and_item(cuisine):
    template = """
    I want to open a restaurant for {cuisine} food. suggest a great name for this.
    """

    prompt = PromptTemplate(input_variables=["cuisine"], template=template)
    # prompt.format(cuisine="indian")

    name_chain = LLMChain(prompt=prompt, llm=llm, output_key="restaurant_name")

    # print(name_chain.run("italian"))

    food_template = """
    suggest some menu items for {restaurant_name}, and return it as comma seperated list.
    """

    food_prompt = PromptTemplate(
        input_variables=["restaurant_name"], template=food_template)

    food_chain = LLMChain(prompt=food_prompt, llm=llm, output_key="menu_items")

    chain = SequentialChain(chains=[name_chain, food_chain], input_variables=[
        "cuisine"], output_variables=["restaurant_name", "menu_items"], verbose=False)  # order of chain matters.

    response = chain({"cuisine": cuisine})
    # result = chain("indian")

    return response


if __name__ == "__main__":
    print(generate_restaurant_name_and_item("indian"))
