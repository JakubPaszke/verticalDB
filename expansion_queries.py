import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from helper_utils import word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import numpy as np
import umap


load_dotenv()

openai_key = os.getenv("API_KEY")
client = OpenAI(api_key=openai_key)

reader = PdfReader("data/logint.pdf")
pdf_texts = [p.extract_text() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256, chunk_overlap=0)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function(token_split_texts[10]))

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("logint-collection", embedding_function=embedding_function)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
count = chroma_collection.count()

query = "What is the purpose of the document?"
results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_docs = results["documents"][0]

def generate_multi_query(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert programming resaearch assistant. 
    Your users are inquiring about systems and they try to look for projects and authors with experience.
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering."""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    content = response.choices[0].message.content
    content = content.split("\n")
    return content

original_query = (
    "How exactly does user interact with LogInt, could you describe process of integration?"
)

aug_queries = generate_multi_query(original_query)

# 1. First step show the augmented queries
for query in aug_queries:
    print("\n", query)

# 2. concatenate the original query with the augmented queries
joint_query = [
    original_query
] + aug_queries  # original query is in a list because chroma can actually handle multiple queries, so we add it in a list

# print("======> \n\n", joint_query)

results = chroma_collection.query(
    query_texts=joint_query, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# # output the results documents
# for i, documents in enumerate(retrieved_documents):
#     print(f"Query: {joint_query[i]}")
#     print("")
#     print("Results:")
#     for doc in documents:
#         print(word_wrap(doc))
#         print("")
#     print("-" * 100)

# Combine all unique documents into one large context string
context_text = "\n".join(unique_documents)

def get_final_answer(query, context, model="gpt-3.5-turbo"):
    """
    Calls OpenAI Chat Completion API to get a final answer based on the retrieved context.
    """
    # Prepare a system prompt or introduction that sets the role or style of the assistant
    system_message = (
        "You are a helpful assistant that uses the provided context to answer the user's question. "
        "Do not halucinate."
    )
    
    # Prepare the user prompt that includes the relevant context
    user_message = f"""
    Here is some context:
    {context}
    
    The user's question is:
    {query}
    
    Please provide a concise, accurate, and direct answer, citing relevant points from the context.
    """
    
    # Create the messages payload
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    # Call the Chat Completion endpoint
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    # Extract the final answer from the API response
    final_answer = response.choices[0].message.content
    return final_answer

# 2) Get a final answer from the LLM
final_answer = get_final_answer(joint_query, context_text)

print("\n=== FINAL ANSWER ===")
print(final_answer)