# load env data
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import pinecone

# Split big PDF in independent chunks.
pdf_loader = PyPDFLoader("./data/field-guide-to-data-science.pdf")
pdf_data = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
texts = text_splitter.split_documents(pdf_data)

# Initialize embedding class
embedding_obj = OpenAIEmbeddings()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_API_ENV"))

index_name = "langchaintest1"

# Embed and store data in vetor store.
embed_data = Pinecone.from_texts([t.page_content for t in texts], embedding=embedding_obj, index_name=index_name)

# Search in vector store for matching documents.
query = "What are the components of a computer?"

filtered_doc = embed_data.similarity_search(query=query)

print(filtered_doc)

print("********** Generate a response by OpenAI **********")

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm=llm, chain_type="stuff")

response = chain.run(input_documents = filtered_doc, question = query)

print(response)



