import os
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv


# load_dotenv(override=True)

# # # Get the API key
# openai_api_key = os.getenv('OPENAI_API_KEY')
# print(openai_api_key)
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Reload the vector store
vectorstore = Chroma(persist_directory="labor_laws_embeddings", embedding_function=embeddings)

# Arabic query
# query = "ما هي الشروط اللازمة لتوظيف العمالة غير السعودية؟"
# query = "ما هي المتطلبات التي يجب توفرها لتوظيف العمالة الأجنبية فقط؟"
# query = "ما هي الشروط اللازمة لتوظيف العمالة غير السعودية أو العمالة الوافدة؟"
# query = "ما هي الشروط اللازمة لتوظيف العمالة السعودية؟"
# query = "أنا موظف عملت لدى صاحب عمل في المملكة العربية السعودية لمدة خمس سنوات بعقد غير محدد المدة. تم إنهاء عقدي مؤخراً بسبب ظروف اقتصادية تمر بها الشركة، وتم إبلاغي قبل شهر واحد فقط. أريد معرفة ما يلي:ما إذا كان إنهاء عقدي قانونياً في ظل هذه الظروف"

# # Perform a similarity search
# results = vectorstore.similarity_search(query, k=3)  # Adjust k for the number of results

# # Save results to a text file
# output_file = "retrieved_results.txt"
# with open(output_file, "w", encoding="utf-8") as file:
#     for i, result in enumerate(results):
#         file.write(f"نتيجة {i + 1}:\n")
#         file.write(f"المحتوى: {result.page_content}\n")
        
# print(f"Results have been saved to {output_file}")
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

generator_embeddings = LangchainEmbeddingsWrapper(vectorstore._embedding_function)

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
# generator_llm = ChatOpenAI(
#     model="gpt-4",
#     temperature=0.7,  # Controls the creativity of responses
#     openai_api_key= openai_api_key
# )
import json 
from langchain_core.documents.base import Document

json_file_path = "labor_laws_chunks.json"  

with open(json_file_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)
documents = [
    Document(page_content=chunk["content"], metadata=chunk["metadata"])
    for chunk in chunks
]
# Initialize TestsetGenerator
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

# Generate test set
print(generator)

# testset = generator.generate_with_langchain_docs(documents, testset_size=1)
# print(2)
# # Save the generated test set
# testset.to_pandas()
# print(3)
