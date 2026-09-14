from flask import Flask, render_template, request
from src.helper import download_huggingface_embeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from src.prompt import system_prompt
from google import genai
import os


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Configure Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

# Load embeddings
embeddings = download_huggingface_embeddings()

# Connect to existing Pinecone index
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():

    msg = request.form["msg"]

    print("Question:", msg)

    # Retrieve relevant medical documents from Pinecone
    documents = retriever.invoke(msg)

    # Combine retrieved documents into context
    context = "\n\n".join(
        document.page_content for document in documents
    )

    # Create prompt for Gemini
    prompt = f"""
{system_prompt}

Context from medical documents:
{context}

User Question:
{msg}

Answer:
"""

    # Generate response using Gemini
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = response.text

    print("Response:", answer)

    return answer


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )