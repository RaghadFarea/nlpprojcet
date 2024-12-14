import json 
import os
from dotenv import load_dotenv
import shutil


load_dotenv()

# Get the API key
openai_api_key = os.getenv('OPENAI_API_KEY')

persist_directory = "labor_laws_embeddings"
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)

json_file_path = "labor_laws_chunks.json"  
with open(json_file_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

# from langchain_ollama import OllamaEmbeddings
from langchain_core.documents.base import Document
from langchain.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=openai_api_key
   
)
documents = [
    Document(page_content=chunk["content"], metadata=chunk["metadata"])
    for chunk in chunks
]

# Create a Chroma vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="labor_laws_embeddings"
)
query = "ما هي الشروط اللازمة لتوظيف العمالة غير السعودية؟"
results = vectorstore.similarity_search(query, k=3)

output_file = "retrieved_results.txt"
with open(output_file, "w", encoding="utf-8") as file:
    for i, result in enumerate(results):
        file.write(f"نتيجة {i + 1}:\n")
        file.write(f"المحتوى: {result.page_content}\n")

print(f"Results have been saved to {output_file}")

# embeddings = OllamaEmbeddings(
#     model="llama3.2"
# )

# # # Optional: Generate embeddings for a sample text to test
# # text = "Sample text to generate embeddings for."
# # embedding_vector = embeddings.embed_query(text)
# # print("Sample Embedding Vector:", embedding_vector)

# documents = [
#     Document(page_content=chunk["content"], metadata=chunk["metadata"])
#     for chunk in chunks
# ]

# # print(documents)

# # Create a Chroma vector store to store embeddings
# vectorstore = Chroma.from_documents(
#     documents=documents,
#     embedding=embeddings,
#     persist_directory="labor_laws_embeddings"  # Directory to persist embeddings
# )

# # Persist the vectorstore
# vectorstore.persist()