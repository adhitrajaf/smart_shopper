# 🛍️ SmartShopper AI Assistant
SmartShopper is an intelligent, conversational shopping assistant designed to enhance the e-commerce experience. Built with a powerful Retrieval-Augmented Generation (RAG) architecture, this application can understand user queries, provide relevant product recommendations with dynamic filtering, and answer frequently asked questions about store policies.

The entire system is served through an interactive and user-friendly web interface created with Streamlit.

✨ Key Features
Conversational Interface: A real-time, chat-based UI powered by Streamlit for a natural and engaging user experience.

Intelligent Product Search: Utilizes semantic search to understand the intent behind user queries and retrieve the most relevant products.

Dynamic Metadata Filtering: The AI can generate filters directly from natural language queries (e.g., "Show me cotton shirts under 500k") to refine search results by category, material, or price.

RAG for FAQ: Instantly answers common questions about shipping, return policies, and payment methods by retrieving information from a dedicated knowledge base.

High-Speed LLM: Powered by the Groq LPU™ Inference Engine for fast and fluid responses.

Scalable Vector Database: Uses MongoDB Atlas as a vector store for efficient document retrieval and semantic search.

🛠️ Technology Stack
Frontend: Streamlit

AI / RAG Framework: Haystack

LLM Provider: Groq (using Llama 3)

Database: MongoDB Atlas (for Vector Search)

Core Libraries: PyMongo, Sentence-Transformers, python-dotenv

🚀 Getting Started
Follow these steps to set up and run the project locally.

1. Prerequisites
Python 3.9+

A MongoDB Atlas account with a running cluster.

A Groq API Key.

2. Installation & Setup
Clone the repository:

Bash

git clone https://github.com/adhitrajaf/smart_shopper.git
cd smart_shopper
Create and activate a virtual environment:

Bash

# Windows
python -m venv smartshopper
.\smartshopper\Scripts\activate

# macOS / Linux
python3 -m venv smartshopper
source smartshopper/bin/activate
Install the dependencies:

Bash

pip install -r requirements.txt
Set up environment variables:
Create a file named .env in the root directory of the project and add your secret keys:

Code snippet

GROQ_API_KEY="gsk_YourGroqApiKey"
MONGO_CONNECTION_STRING="mongodb+srv://YourMongoURI"
3. Populate the Database
Before running the main application, you must populate your MongoDB Atlas database with product data, embeddings, and search indexes.

Open and run the Jupyter Notebook located at process/store_data.ipynb. This script will create sample collections, generate vector embeddings, and store everything in your cluster.

4. Run the Application
Before run the main website.py u should run jupyter notebook store_data.ipynb first to get data and connect to MongoDB
Once the database is populated, you can start the Streamlit web application.

Bash

streamlit run website.py
Open your web browser and navigate to http://localhost:8501.

📁 Project Structure
.
├── process/
│   ├── store_data.ipynb          # IMPORTANT: Run this first to set up the database.
│   ├── generator.ipynb           # Notebook for testing the RAG generator.
│   ├── generator_filter.ipynb    # Notebook for testing the filter generator.
│   ├── retriever.ipynb           # Notebook for testing the retrieval component.
│   └── ...                       # Other development notebooks.
├── .env                          # Stores secret keys (API, DB connection string).
├── .gitignore                    # Specifies files for Git to ignore.
├── requirements.txt              # Lists all Python dependencies for the project.
├── template.py                   # Contains prompt templates for the LLMs.
└── website.py                    # The main Streamlit application file.
