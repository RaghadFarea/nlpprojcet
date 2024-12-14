# Huquqi: Saudi Arabic Chatbot for Labor Legal Guidance

# 1. Prerequisites
1- install llama 3.2 from Ollama
2- Git a key from OpenAI

# 2. How to run it locally

## 1.Environment Variables

Make sure to create a .env file in the project root with the following variables:

OPENAI_API_KEY == < Your_OPENAI_API_KEY>

## 2. Required folders
1- labor_laws_embeddings
1- my_testset3.jsonl


## 3. Set up a virtual environment:
python -m venv venv
venv\Scripts\activate  

## 4. Install the dependencies:
pip install -r requirements.txt

## 5. Run evaluation.py file:

python evaluation.py


# Important Note

You don't need to run the rest of the files, because we used them to prepare the data and to try to speed up the retrieval process we prepared and stored the embeddings for the chunks so we could just call them up.
