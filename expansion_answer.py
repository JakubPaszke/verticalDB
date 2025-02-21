from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import (RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import umap
import matplotlib.pyplot as plt 

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

# print(word_wrap(character_split_texts[10]))
# print(f"\nTotal chunks: {len(character_split_texts)}")

token_splitter = SentenceTransformersTokenTextSplitter(tokens_per_chunk=256, chunk_overlap=0)

token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# print(word_wrap(token_split_texts[10]))
# print(f"\nTotal chunks: {len(token_split_texts)}")

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

# for document in retrieved_docs:
#     print(word_wrap(document))
#     print("\n")


def augmented_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful expert programming resaearch assistant. 
    Provide an example answer to the given question, that might be present in a document like LogInt."""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content  
    return content

original_query = "What is the purpose of the LogInt system?"
hypothetical_answer = augmented_query_generated(original_query)

joint_query = f"{original_query}\n\n{hypothetical_answer}"
# print(word_wrap(joint_query))

results = chroma_collection.query(query_texts=[joint_query], n_results=5, include=["documents", "embeddings"])
retrieved_docs = results["documents"][0]

# for document in retrieved_docs:
#     print(word_wrap(document))
#     print("\n")

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=15, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrived_embeddings = results["embeddings"][0]
original_retrieved_embeddings = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(original_retrieved_embeddings, umap_transform)
projected_augmented_query_embedding = project_embeddings(augmented_query_embedding, umap_transform)
projected_retrieved_embeddings = project_embeddings(retrived_embeddings, umap_transform)

plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="x",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="x",
    color="orange",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.savefig('plot.png')
 
