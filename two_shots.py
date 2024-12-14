import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)





load_dotenv()

# Get the API key
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Reload the vector store
vectorstore = Chroma(persist_directory="labor_laws_embeddings", embedding_function=embeddings)



retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)

query = "أنا موظف عملت لدى صاحب عمل في المملكة العربية السعودية لمدة خمس سنوات بعقد غير محدد المدة. تم إنهاء عقدي مؤخراً بسبب ظروف اقتصادية تمر بها الشركة، وتم إبلاغي قبل شهر واحد فقط. أريد معرفة ما يلي:ما إذا كان إنهاء عقدي قانونياً في ظل هذه الظروف"


llm = ChatOpenAI(model="gpt-4"  ,  temperature=0,
)
llm2 = ChatOllama(
    model="llama3.1",
    temperature=0,
)


# Define the examples we'd like to include (for 1-shot prompt) with GPT
examples = [
    {
        "input": "عملت ساعات إضافية في وظيفتي ولم أحصل على أجر إضافي. ما هي حقوقي؟",
        "context": """
        المادة 107 من نظام العمل في المملكة العربية السعودية تنص على ما يلي:
        1. يجب أن يدفع للعامل عن ساعات العمل الإضافية أجر الساعة العادية مضافًا إليه 50% من الأجر الأساسي.
        2. تُعتبر الساعات الإضافية هي الساعات التي يعملها العامل بعد ساعات العمل اليومية أو الأسبوعية النظامية المحددة في عقد العمل أو في نظام العمل.
        """,
        "output": """
        وفقًا للمادة 107 من نظام العمل، يستحق العامل أجرًا عن ساعات العمل الإضافية يساوي أجر الساعة العادي مضافًا إليه 50% من الأجر الأساسي كتعويض عن العمل الإضافي.
        """
    },
    {
        "input": "تم فصلي من العمل دون سبب واضح، كيف يمكنني المطالبة بحقي؟",
        "context": """
        المادة 80 من نظام العمل في المملكة العربية السعودية تنص على ما يلي:
        1. لا يجوز لصاحب العمل إنهاء خدمة العامل دون سبب مشروع، إلا في حالات محددة مثل الغش أثناء التوظيف، أو الإخلال بالواجبات الجوهرية للعامل.
        2. في حال تعرض العامل للفصل التعسفي، يمكنه تقديم شكوى إلى مكتب العمل للمطالبة بحقه.
        3. على صاحب العمل إثبات أن الفصل تم بناءً على أسباب مشروعة.
        """,
        "output": """
        إذا تعرض العامل لإنهاء خدمة غير مبرر، يمكنه تقديم شكوى إلى مكتب العمل. ووفقًا للمادة 80، لا يجوز لصاحب العمل إنهاء خدمة العامل دون سبب مشروع، إلا في حالات محددة مثل الغش في التوظيف أو الإخلال بالواجبات الجوهرية.
        """
    }
]


# prompt template to format each individual example
example_prompt = ChatPromptTemplate.from_messages(
     [
        ("human", "Question: {input}\nContext: {context}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


system_prompt = (
    "You are a legal assistant specializing in Saudi labor law. Your role is to answer specific legal questions related to worker rights and labor law in Saudi Arabia. You will retrieve answers based on the case details provided by the user and respond directly to their question. If you cannot find relevant information or if the answer is unclear, you should respond with 'I don't know.'"    
    "Use the given context to answer the question."
    "Use three sentences maximum and keep the answer concise."
    "Context: {context}"
)
# assemble final prompt and use it with a model
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, final_prompt)
chain = create_retrieval_chain(vectorstore.as_retriever(), question_answer_chain)

result = chain.invoke({"input": query})
# Extract the response text from the result dictionary


response_text = str(result.get('answer'))

output_file = "query_result.txt"
with open(output_file, "w", encoding="utf-8") as file:
    file.write(f"Query: {query}\n\n")
    file.write("Result:\n")
    file.write(response_text)

print(f"The result has been written to {output_file}")
