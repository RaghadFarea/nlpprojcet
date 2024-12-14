import os
import giskard
import pandas as pd
import json 
from giskard.rag import generate_testset, KnowledgeBase
from giskard.rag.question_generators import (
    simple_questions,
    complex_questions,
    distracting_questions,
    double_questions,
)
# Get the API key

with open("labor_laws_chunks.json", "r", encoding="utf-8") as file:
    data = json.load(file)
rows = [
    {
        "content": item["content"],
        "باب": item["metadata"].get("باب", ""),
        "فصل": item["metadata"].get("فصل", ""),
        "مادة": item["metadata"].get("مادة", ""),
        "jurisdiction": item["metadata"].get("jurisdiction", "")
    }
    for item in data
]

df = pd.DataFrame(rows)

# Step 3: Create a KnowledgeBase
knowledge_base = KnowledgeBase.from_pandas(df, columns=["content","فصل", "باب"])

# # Verify the knowledge base
print(knowledge_base)
openai_api_key = os.getenv('OPENAI_API_KEY')

os.environ["OPENAI_API_KEY"] = openai_api_key

giskard.llm.set_llm_model("gpt-4o")
giskard.llm.set_embedding_model("text-embedding-3-small")




print(1)

# Generate a testset with 10 questions & answers for each question types (this will take a while)
testset = generate_testset(
    knowledge_base,
    num_questions=60,
      question_generators=[
        simple_questions,  # Include simple questions
        complex_questions,  # Include complex questions
        distracting_questions,  # Include distracting questions
        double_questions,  # Include double questions
    ],
    language='ar',  
    agent_description = ("مساعد قانوني للإجابة على أسئلة تتعلق بأنظمة العمل في السعودية. يوفر الإجابات بناءً على المواد القانونية المسترجعة."

)
)
print(2)
# Save the generated testset
testset.save("my_testset3.jsonl")

# # You can easily load it back
# from giskard.rag import QATestset

# loaded_testset = QATestset.load("my_testset.jsonl")


# # Convert it to a pandas dataframe
# df = loaded_testset.to_pandas()ساعد قانوني