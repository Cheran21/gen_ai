import os
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_charan_krithi = user_secrets.get_secret("API_CHARAN_KRITHI")
os.environ["API_CHARAN_KRITHI"]=api_charan_krithi

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(model_name="llama-3.3-70b-versatile",api_key = api_charan_krithi)
llm

import fitz

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file and returns it as a string."""
    doc = fitz.open(pdf_path)
    text=""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

pdf_text = extract_text_from_pdf("/kaggle/input/indian-penal-code/ipc.pdf")
print("Extracted text from IPC PDF:",len(pdf_text),"characters")

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def create_faiss_index(text):
    """Chunks IPC text and stores embeddings in FAISS."""
    text_splitter =  RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50)
    texts = text_splitter.split_text(text)
    embeddings = hf_model.encode(texts)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(np.array(embeddings))
    return index, texts

ipc_faiss_index, ipc_chunks = create_faiss_index(pdf_text)
print("FAISS Index created with", len(ipc_chunks), "chunks")    

def retrieve_ipc_section(query):
    """FInd the most relevant IPC Section based on the user query."""
    query_embedding = hf_model.encode([query])
    distances, indices = ipc_faiss_index.search(np.array(query_embedding), k=1)
    return ipc_chunks[indices[0][0]] if indices[0][0] < len(ipc_chunks) else "no relevant section"

query = "What is the punishment for theft under IPC"
retrieved_section = retrieve_ipc_section(query)
print("\nRelevant IPC Section:\n",retrieved_section )

prompt = PromptTemplate(
    input_variables=["ipc_section",query],
    template="""
    You are an expert in Indian law. A user asked:"{query}"
    Based on the Indian Penal Code(IPC), th relevant section is :
    {ipc_section}

    please provide:
    -A simple explanation
    -the key legal points
    -possible punishments
    -a real-world example
    """
)
def query_groq(prompt):
    responses = chain.run()
    print(responses)
    return responses

def ipc_chatbot(query):
    releated_section = retrieve_ipc_section(query)
    chain=LLMChain(llm=llm, prompt=prompt)
    responses = chain.run(ipc_section=releated_section,query=query)
    return responses

user_query = input("Enter your leagal question:")
chatbot_response = ipc_chatbot(user_query)
print(chatbot_response)
