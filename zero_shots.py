import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Get the API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Reload the vector store
vectorstore = Chroma(persist_directory="labor_laws_embeddings", embedding_function=embeddings)

query = "ما الذي يحظره النظام على العامل؟"

# Perform a similarity search
retrieved_docs = vectorstore.similarity_search(query, k=1)  # Retrieve the top 3 most similar documents

# Initialize LLM
llm2 = ChatOpenAI(model="gpt-4", temperature=0)

# Define a system prompt
system_prompt = (
    "You are a legal assistant specializing in Saudi labor law. Your role is to answer specific legal questions related to worker rights and labor law in Saudi Arabia. "
    "You will retrieve answers based on the case details provided by the user and respond directly to their question. If you cannot find relevant information or if the answer is unclear, "
    "you should respond with 'I don't know.' Use the given context to answer the question. Use three sentences maximum and keep the answer concise. Context: {context}"
)

# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the retrieval and question-answering chain
question_answer_chain = create_stuff_documents_chain(llm2, prompt)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chain = create_retrieval_chain(retriever, question_answer_chain)

# Get the answer from the chain
result = chain.invoke({"input": query})
response_text = result.get("answer", "No answer provided.")

# Save the query, retrieved contexts, and response to a file
output_file = "query_result.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(f"Query: {query}\n\n")
    file.write("Retrieved Documents:\n")
    for i, doc in enumerate(retrieved_docs, start=1):
        file.write(f"Document {i}: {doc.page_content}\n\n")
    file.write("Result:\n")
    file.write(response_text)

print(f"The result and retrieved documents have been written to {output_file}")
