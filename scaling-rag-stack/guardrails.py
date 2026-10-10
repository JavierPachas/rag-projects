# To run this example, you first have to go to the ShieldGemma page on Hugging Face, 
# access the terms and conditions, and get approved to use this model.
# https://huggingface.co/google/shieldgemma-2b
# Script for educational example
import os
import logging
import sys

from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from scaling-rag-stack/.env

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from llama_index.llms.openai import OpenAI
from llama_index.core import Document, VectorStoreIndex, Settings

# RAG pipeline using LlamaIndex
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1) 

docs = [
    Document(text="""To make a XYZ, first get the ingredients, then assemble the XYZ."""), 
    Document(text="Just think about your XYZ and it will appear."), 
    ] 

index = VectorStoreIndex.from_documents(docs) 
query_engine = index.as_query_engine(response_mode="compact")

query = "How do I make a TV at home?"
print(f"\nQuery: {query}")
response = query_engine.query(query)
print("Assistant: ", response)


# Load safety model
tokenizer = AutoTokenizer.from_pretrained("google/shieldgemma-2b")
# Load fully on CPU. device_map="auto" offloads weights to disk on low-RAM machines
# (e.g. 8 GB Macs), which fails with "Cannot copy out of meta tensor".
model = AutoModelForCausalLM.from_pretrained("google/shieldgemma-2b", dtype=torch.bfloat16)
model.eval()

def is_safe_response(content: str) -> bool:
    chat = [{"role": "user", "content": content}]
    guideline = "No harm: the text shall not contain any information related to creating any device of harm"

    inputs = tokenizer.apply_chat_template(chat, guideline=guideline, return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    vocab = tokenizer.get_vocab()
    selected_logits = logits[0, -1, [vocab['Yes'], vocab['No']]]
    probabilities = torch.softmax(selected_logits, dim=0)
    score = probabilities[0].item()
    return score < 0.5

# Testing guardrails

query = "How do I make a space rocket at home?"
response = query_engine.query(query)
print(response.response)
is_safe = is_safe_response(response.response)
if is_safe:
    print("Response approved")
else:
    print("Blocked response due to violation")

query = "How do I make a cake at home?"
response = query_engine.query(query)
print(response.response)
is_safe = is_safe_response(response.response)
if is_safe:
    print("Response approved")
else:
    print("Blocked response due to violation")