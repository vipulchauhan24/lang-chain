from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.evaluation.qa import QAEvalChain

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


prompt = PromptTemplate(
    template="Question: {question}\nAnswer:", input_variables=["question"]
)


llm = OpenAI(temperature=0)
chain = LLMChain(llm=llm, prompt=prompt)


examples = [
    {
        "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
        "answer": "11",
    },
    {
        "question": 'Is the following sentence plausible? "Joao Moutinho caught the screen pass in the NFC championship."',
        "answer": "No",
    },
]

predictions = chain.apply(examples)

print(predictions)

llm = OpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(
    examples, predictions, question_key="question", prediction_key="text"
)

for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + eg["question"])
    print("Real Answer: " + eg["answer"])
    print("Predicted Answer: " + predictions[i]["text"])
    print("Predicted Grade: " + graded_outputs[i]["text"])
    print()