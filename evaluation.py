import os
from dotenv import load_dotenv
import json
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from giskard.rag import evaluate, QATestset, AgentAnswer
from giskard.rag.metrics.ragas_metrics import (
    ragas_context_precision,
    ragas_faithfulness,
    ragas_answer_relevancy,
    ragas_context_recall
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# Load environment variables
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')


# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Reload the vector store
vectorstore = Chroma(persist_directory="labor_laws_embeddings", embedding_function=embeddings)

# Create retriever with a similarity threshold
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  # Retrieve only 1 example, no threshold
)
# Initialize LLM
# llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

#think is for few shots
# examples = [
#     {
#      "input": "ما هي الظروف التي تؤدي إلى انتهاء عقد العمل؟",
#     "context": """
#     Document 94: content: لا ينقضي عقد العمل بوفاة صاحب العمل، ما لم تكن شخصيته قد روعيت في إبرام العقد ولكنه ينتهي بوفاة العامل أو بعجزه عن أداء عمله، وذلك بموجب شهادة طبية معتمدة من الجهات الصحية المخولة أو من الطبيب المخول الذي يعينه صاحب العمل.
#     باب: الباب الخامس: علاقات العمل
#     فصل: الفصل الثالث: انتهاء عقد العمل
#     """,
#     "output": """
#     ينتهي عقد العمل في حال وفاة العامل أو عجزه عن أداء عمله، وفقًا لشهادة طبية معتمدة من الجهات الصحية المخولة أو من الطبيب الذي يعينه صاحب العمل.
#     """
#     },

#     {
#     "input": "ما هي معايير تحديد المنشآت الخطرة؟",
#     "context": """
#     Document 154: content: تضع الوزارة ضوابط لتحديد "المنشآت ذات المخاطر الكبرى" استناداً إلى قائمة المواد الخطرة، أو فئات هذه المواد أو كلتيهما.
#     باب: الباب الثامن: الوقاية من مخاطر العمل والوقاية من الحوادث الصناعية الكبرى وإصابات العمل والخدمات الصحية والاجتماعية
#     فصل: الفصل الثاني:الوقاية من الحوادث الصناعية الكبرى
#     """,
#     "output": """
#     تضع الوزارة ضوابط لتحديد المنشآت الخطرة بناءً على قائمة المواد الخطرة، أو فئات هذه المواد، أو كلاهما معًا.
#     """
# },

# {
#     "input": "ما هي المعلومات الأساسية التي يجب أن يتضمنها عقد العمل في النظام السعودي؟",
#     "context": """
#     Document 65: content: 1. مع مراعاة ما ورد في المادة (السابعة والثلاثين) من هذا النظام، تضع الوزارة نموذجاً موحداً لعقد العمل، يحتوي بصورة أساسية على: اسم صاحب العمل ومكانه، واسم العامل وجنسيته، وما يلزم لإثبات شخصيته، وعنوان إقامته، والأجر المتفق عليه بما في ذلك المزايا والبدلات، ونوع العمل ومكانه، وتاريخ الالتحاق به، ومدته إن كان محدد المدة.
#     2. يجب أن يكون عقد العمل وفق النموذج المشار إليه في الفقرة (1) من هذه المادة، ولطرفي العقد أن يضيفا إليه بنوداً أخرى، بما لا يتعارض مع أحكام هذا النظام ولائحته والقرارات الصادرة تنفيذاً له.
#     باب: الباب الخامس: علاقات العمل
#     فصل: الفصل الأول: عقد العمل
#     """,
#     "output": """
#     يجب أن يتضمن عقد العمل اسم صاحب العمل ومكانه، واسم العامل وجنسيته، وعنوانه، والأجر المتفق عليه، ونوع العمل ومكانه، وتاريخ الالتحاق به، ومدته إذا كان محدد المدة.
#     """
# }

# ]
# # prompt template to format each individual example
# example_prompt = ChatPromptTemplate.from_messages(
#      [
#         ("human", "Question: {input}\nContext: {context}"),
#         ("ai", "{output}"),
#     ]
# )
# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=example_prompt,
#     examples=examples,
# )


# system_prompt = (
#     "You are a legal assistant specializing in Saudi labor law. Your role is to answer specific legal questions related to worker rights and labor law in Saudi Arabia. "
#     "You are provided with examples of how to answer similar questions, which include a question, a context, and an output. These examples are intended to guide you in structuring your responses. "
#     "Use the given context to answer the question directly, following the style and structure demonstrated in the examples. "
#     "If you cannot find relevant information or if the answer is unclear, respond with 'I don't know.' "
#     "Keep your answer concise and use no more than three sentences. "
#     "Context: {context}"
# )
# # assemble final prompt and use it with a model
# final_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system",system_prompt),
#         few_shot_prompt,
#         ("human", "{input}"),
#     ]
# )
# Define a system prompt
system_prompt = (
    "You are a legal assistant specializing in Saudi labor law. Your role is to answer specific legal questions related to worker rights and labor law in Saudi Arabia. "
    "You will retrieve answers based on the case details provided by the user and respond directly to their question. If you cannot find relevant information or if the answer is unclear, "
    "you should respond with 'I don't know.' Use the given context to answer the question. Use three sentences maximum and keep the answer concise. Context: {context}"
)
# system_prompt = (
#     "You are a legal assistant specializing in Saudi labor law. Your role is to answer specific legal questions related to worker rights and labor law in Saudi Arabia. "
#     "You will retrieve answers based on the case details provided by the user to their question. if the answer is unclear, "
#     "you should respond with 'I don't know.' Use the given context to answer the question. Use four sentences maximum and keep the answer concise. Context: {context}"
# )
# Create a chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Combine documents and create the chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

def get_answer_fn(question: str, history=None) -> AgentAnswer:
    """Function to get the RAG agent's answer and retrieved documents."""
    # Format history (if applicable)
    messages = history if history else []
    messages.append({"role": "user", "content": question})

    # Retrieve documents explicitly using the retriever
    retrieved_docs = retriever.get_relevant_documents(question)

    # Extract context as a string
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Get the answer from the chain
    result = chain.invoke({"input": question, "context": context})
    answer = result.get("answer", "No answer provided.")

    # Return the AgentAnswer object with string context
    return AgentAnswer(
        message=answer,
        documents=[doc.page_content for doc in retrieved_docs]  # Ensure this is a list of strings
    )

# Load the test set from JSONL
file_path = "my_testset3.jsonl"
testset = QATestset.load(file_path)

# Evaluate the pipeline
report = evaluate(
    get_answer_fn,
    testset=testset,
    metrics=[ragas_context_precision, ragas_faithfulness, ragas_answer_relevancy, ragas_context_recall]
)

# Save as HTML
report.to_html("evaluation_results2_3k_llama_0shots.html")

# Save results as a DataFrame
results_df = report.to_pandas()
results_df.to_csv("evaluation_results2_3k_llama_0shots.csv", index=False, encoding="utf-8")

import pandas as pd

# Load evaluation results from the CSV file
results_df = pd.read_csv("evaluation_results2_3k_llama_0shots.csv")

# Calculate the overall score for each metric
overall_scores = {
    "Context Precision": results_df["RAGAS Context Precision"].mean(),
    "Faithfulness": results_df["RAGAS Faithfulness"].mean(),
    "Answer Relevancy": results_df["RAGAS Answer Relevancy"].mean(),
    "Context Recall": results_df["RAGAS Context Recall"].mean()
}

# Display the overall scores
for metric, score in overall_scores.items():
    print(f"{metric}: {score:.4f}")



import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load test set results
results_df = pd.read_csv("evaluation_results2_3k_llama_0shots.csv")

# Prepare data for BLEU evaluation
questions = []
references = []
responses = []
bleu_scores = []

# Smoothing function for BLEU score
smoothing_function = SmoothingFunction().method2

# Ensure the DataFrame contains the required columns
if not {"question", "reference_answer", "agent_answer"}.issubset(results_df.columns):
    raise ValueError("Input DataFrame must contain 'question', 'reference_answer', and 'agent_answer' columns.")

for _, row in results_df.iterrows():
    question = row["question"]
    reference = row["reference_answer"]
    response = row["agent_answer"]

    # Calculate BLEU score
    bleu_score = sentence_bleu(
        [reference.split()],  # Reference answer as a list of words
        response.split(),     # Generated answer as a list of words
        smoothing_function=smoothing_function
    )

    # Store results
    questions.append(question)
    references.append(reference)
    responses.append(response)
    bleu_scores.append(bleu_score)

# Calculate overall BLEU score
overall_bleu_score = sum(bleu_scores) / len(bleu_scores)

# Display results
print(f"Overall BLEU Score: {overall_bleu_score:.4f}")

# Save to CSV for further analysis
output_df = pd.DataFrame({
    "Question": questions,
    "Reference Answer": references,
    "Agent Answer": responses,
    "BLEU Score": bleu_scores
})
output_df.to_csv("nltk_bleu_scores.csv", index=False, encoding="utf-8")

print("BLEU scores have been saved to 'nltk_bleu_scores.csv'.")

from bert_score import score

# Load the evaluation results
results_df = pd.read_csv("evaluation_results2_3k_llama_0shots.csv")

# Ensure the DataFrame contains the required columns
if not {"reference_answer", "agent_answer"}.issubset(results_df.columns):
    raise ValueError("Input DataFrame must contain 'reference_answer' and 'agent_answer' columns.")

# Prepare references and candidates
references = results_df["reference_answer"].tolist()
candidates = results_df["agent_answer"].tolist()

# Calculate BERTScore
P, R, F1 = score(candidates, references, lang="ar")  # Use lang="ar" for Arabic

# Add BERTScore results to the DataFrame
results_df["BERT Precision"] = P.tolist()
results_df["BERT Recall"] = R.tolist()
results_df["BERT F1"] = F1.tolist()

# Save the updated DataFrame with BERTScore
results_df.to_csv("bert_score_results.csv", index=False, encoding="utf-8")

# Print overall BERT F1
overall_bert_f1 = F1.mean()
print(f"Overall BERTScore-F1: {overall_bert_f1:.4f}")
