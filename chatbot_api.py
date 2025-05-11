
# chatbot_api.py

import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# LangChain & OpenAI imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set your OpenAI key here or via environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "sk-...")

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load tourism-related URLs
urls = [
    "https://www.visitsaudi.com/en",
    "https://en.wikipedia.org/wiki/Tourism_in_Saudi_Arabia",
    "https://www.lonelyplanet.com/saudi-arabia",
    "https://www.bucketlistly.blog/posts/best-places-to-visit-in-saudi-arabia",
    "https://www.visitsaudi.com/en/destinations",
    "https://www.frasershospitality.com/en/saudi-arabia/riyadh/fraser-suites-riyadh/city-guide/attractions-places-to-visit-in-riyadh/",
    "https://www.tripadvisor.com/Restaurants-g293991-Saudi_Arabia.html",
    "https://www.tripadvisor.com/Restaurants-g293995-Riyadh_Riyadh_Province.html",
    "https://www.timeoutriyadh.com/restaurants",
    "https://welcomesaudi.com/restaurant",
    "https://www.visitsaudi.com/en/getting-around"
]

# Load and prepare documents
print("🔁 Loading and processing documents...")
loader = WebBaseLoader(urls)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter()
docs_split = splitter.split_documents(docs)

embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs_split, embedding)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 20}
)

# Define prompt
prompt = PromptTemplate.from_template("""
You are a tourism expert in Saudi Arabia. Your job is to provide accurate and specific answers to tourist's queries.
If they ask about food options, reply with a list of restaurant recommendations.
If they ask about transportation, guide them about transportation.
If they ask about local culture, inform them about the culture.
And so on. Respond to their queries specifically and do not overload them with unnecessary information.

Use the following context to somewhat guide your answer, but do not rely on it exclusively.
If the context is incomplete, feel free to use your own knowledge of Saudi Arabia to answer the question as accurately as possible.
Use line breaks and bullet points to structure the answers neatly.

Context:
{context}

Question: {question}

Answer:
""")

llm = ChatOpenAI(model_name="gpt-4o", max_tokens=1024)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

def get_response(user_input):
    result = rag_chain(user_input)
    return result["result"]

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.get_json().get("message", "")
    reply = get_response(user_input)
    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
