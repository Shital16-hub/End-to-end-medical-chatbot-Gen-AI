from flask import Flask, render_template, request, jsonify
from src.helper import (
    get_embeddings,
    initialize_vectorstore,
    setup_llm,
    create_rag_chain
)
from src.prompt import create_qa_prompt
import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Set Pinecone API key
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables!")

# Initialize Pinecone
Pinecone(api_key=PINECONE_API_KEY)

app = Flask(__name__)

# Initialize components
embeddings = get_embeddings()
vectorstore = initialize_vectorstore("medicalbot", embeddings)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
llm_engine = setup_llm()
prompt = create_qa_prompt()
rag_chain = create_rag_chain(retriever, llm_engine, prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        user_query = request.json['query']
        response = rag_chain.invoke({"input": user_query})
        return jsonify({
            "status": "success",
            "answer": response["answer"]
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)