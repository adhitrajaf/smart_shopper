import streamlit as st
import os
import json
import re
from dotenv import load_dotenv
from haystack import Pipeline, component
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.agents import Agent
from haystack.components.builders import ChatPromptBuilder, PromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.tools.tool import Tool
from haystack_integrations.document_stores.mongodb_atlas import MongoDBAtlasDocumentStore
from haystack_integrations.components.retrievers.mongodb_atlas import MongoDBAtlasEmbeddingRetriever
from haystack_experimental.chat_message_stores.in_memory import InMemoryChatMessageStore
from haystack_experimental.components.retrievers import ChatMessageRetriever
from haystack_experimental.components.writers import ChatMessageWriter
from pymongo import MongoClient
from groq import Groq
from functools import partial
from typing import List, Annotated
import time

# Load environment variables
load_dotenv()

def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #4facfe;
        --accent-secondary: #00f2fe;
        --success-color: #28a745;
        --error-color: #dc3545;
        --warning-color: #ffc107;
        --info-color: #17a2b8;
        --light-bg: #f8fafc;
        --white: #ffffff;
        --text-primary: #2d3748;
        --text-secondary: #4a5568;
        --border-color: #e2e8f0;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Global font settings */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    /* Main container styling */
    .main {
        padding: 1rem 2rem;
        background-color: var(--light-bg);
    }
    
    /* Header styling with better contrast */
    .header-container {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        color: var(--white);
        text-align: center;
        box-shadow: var(--shadow-lg);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        color: var(--white) !important;
    }
    
    .header-subtitle {
        font-size: 1.4rem;
        font-weight: 400;
        opacity: 0.95;
        color: var(--white) !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    /* Chat container styling */
    .chat-container {
        background: var(--white);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    /* Sidebar styling with improved contrast */
    .sidebar-header {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--accent-secondary) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: var(--white);
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        border: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .sidebar-header h3 {
        color: var(--white) !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar-header p {
        color: var(--white) !important;
        opacity: 0.95;
        margin: 0;
        font-weight: 400;
    }
    
    /* Feature box with better visibility */
    .feature-box {
        background: var(--white);
        border-left: 4px solid var(--primary-color);
        padding: 1rem 1.2rem;
        margin: 0.8rem 0;
        border-radius: 8px;
        box-shadow: var(--shadow);
        color: var(--text-primary) !important;
        font-weight: 500;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .feature-box:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        border-left-color: var(--accent-color);
    }
    
    /* Sample question styling */
    .sample-question {
        background: linear-gradient(135deg, #e3f2fd 0%, #f1f8ff 100%);
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid var(--info-color);
        color: var(--text-primary) !important;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(23, 162, 184, 0.2);
    }
    
    /* Status indicators with high contrast */
    .status-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724 !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--success-color);
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: var(--shadow);
        border: 1px solid rgba(40, 167, 69, 0.2);
    }
    
    .status-error {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24 !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--error-color);
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: var(--shadow);
        border: 1px solid rgba(220, 53, 69, 0.2);
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: var(--white);
        border-radius: 10px;
        box-shadow: var(--shadow);
    }
    
    .loading-text {
        margin-left: 1rem;
        font-style: italic;
        color: var(--primary-color) !important;
        font-weight: 500;
    }
    
    /* Button styling improvements */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: var(--white) !important;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
        background: linear-gradient(135deg, var(--secondary-color) 0%, var(--primary-color) 100%);
    }
    
    /* Sidebar button styling - improved visibility */
    .element-container .stButton > button {
        width: 100%;
        text-align: left;
        background: var(--white) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--border-color) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 1rem !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        transition: all 0.3s ease !important;
        margin-bottom: 0.5rem !important;
    }
    
    .element-container .stButton > button:hover {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: var(--white) !important;
        border-color: var(--primary-color) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.2) !important;
    }
    
    .element-container .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Sample question styling for better visibility */
    .sample-question-button {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
        color: var(--text-primary) !important;
        border: 2px solid var(--primary-color) !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }
    
    .sample-question-button:hover {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
        color: var(--white) !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: var(--white);
        border-radius: 10px;
        box-shadow: var(--shadow);
        border: 1px solid var(--border-color);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid var(--border-color);
        padding: 0.75rem;
        font-family: 'Poppins', sans-serif !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Spinner customization */
    .stSpinner {
        color: var(--primary-color) !important;
    }
    
    /* Markdown content improvements */
    .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background-color: var(--white);
    }
    
    /* Override any default text colors */
    .element-container, .stMarkdown, .stText {
        color: var(--text-primary) !important;
    }
    
    /* Ensure all text elements have proper contrast */
    * {
        color: inherit;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--light-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced Templates
METADATA_FILTER_TEMPLATE = """
You are an expert assistant that helps create metadata filters for product searches. 
Based on the user input, create a JSON filter object that can be used to filter products.

Available materials: {{materials}}
Available categories: {{categories}}

The filter should follow this structure:
```json
{
    "material": ["material1", "material2"],  // Only if user mentions specific materials
    "category": ["category1", "category2"],  // Only if user mentions specific categories
    "price": {"$gte": min_price, "$lte": max_price}  // Only if user mentions price range
}
```

Rules:
1. Only include filters that are explicitly mentioned or strongly implied in the user input
2. For materials and categories, use exact matches from the available lists
3. For price, extract numerical values and create range filters
4. If no specific filters are mentioned, return an empty JSON object: {}
5. Always return valid JSON wrapped in ```json``` code blocks

User input: {{input}}

Filter:
"""

# Enhanced Components with proper decorators
@component
class GroqChatGenerator:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model
    
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], tools: List = None):
        if not messages:
            raise ValueError("The 'messages' list received by GroqChatGenerator is empty.")
            
        groq_messages = []
        for msg in messages:
            if hasattr(msg, 'role') and hasattr(msg, 'text'):
                role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
                groq_messages.append({"role": role, "content": msg.text})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return {
                "replies": [
                    ChatMessage.from_assistant(response.choices[0].message.content)
                ]
            }
        except Exception as e:
            st.error(f"Error with Groq API: {e}")
            return {
                "replies": [
                    ChatMessage.from_assistant("I apologize, but I'm experiencing technical difficulties. Please try again.")
                ]
            }

@component
class GroqGenerator:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model
    
    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        if not prompt:
            raise ValueError("The 'prompt' received by GroqGenerator is empty.")
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )
            
            return {
                "replies": [response.choices[0].message.content]
            }
        except Exception as e:
            st.error(f"Error with Groq API: {e}")
            return {
                "replies": ["Error processing request. Please try again."]
            }

class MongoDBAtlas:
    def __init__(self, mongo_connection_string: str):
        self.client = MongoClient(mongo_connection_string)
        self.db = self.client.smartshopper_store
        self.material_collection = self.db.materials
        self.category_collection = self.db.categories

    def get_materials(self):
        try:
            return [doc['name'] for doc in self.material_collection.find()]
        except Exception as e:
            st.error(f"Error fetching materials: {e}")
            return []

    def get_categories(self):
        try:
            return [doc['name'] for doc in self.category_collection.find()]
        except Exception as e:
            st.error(f"Error fetching categories: {e}")
            return []

@component
class GetMaterials:
    def __init__(self):
        self.db = MongoDBAtlas(os.environ['MONGO_CONNECTION_STRING'])
    
    @component.output_types(materials=List[str])
    def run(self):
        materials = self.db.get_materials()
        return {"materials": materials}

@component
class GetCategories:
    def __init__(self):
        self.db = MongoDBAtlas(os.environ['MONGO_CONNECTION_STRING'])
    
    @component.output_types(categories=List[str])
    def run(self):
        categories = self.db.get_categories()
        return {"categories": categories}

class ParaphraserPipeline:
    def __init__(self, chat_message_store):
        self.memory_retriever = ChatMessageRetriever(chat_message_store)
        self.pipeline = Pipeline()
        self.pipeline.add_component("prompt_builder", ChatPromptBuilder(
            variables=["query", "memories"],
            required_variables=["query", "memories"],
        ))
        self.pipeline.add_component("generator", GroqChatGenerator())
        self.pipeline.add_component("memory_retriever", self.memory_retriever)

        self.pipeline.connect("prompt_builder.prompt", "generator.messages")
        self.pipeline.connect("memory_retriever", "prompt_builder.memories")
    
    def run(self, query):
        messages = [
            ChatMessage.from_system("You are a helpful assistant that paraphrases user queries based on previous conversations."),
            ChatMessage.from_user("""
                Please paraphrase the following query based on the conversation history. 
                If the conversation history is empty, please return the query as is.
                
                History:
                {% for memory in memories %}
                    {{memory.text}}
                {% endfor %}
                
                Query: {{query}}
                Answer:
                """)
        ]

        try:
            res = self.pipeline.run(
                data={"prompt_builder": {"query": query, "template": messages}},
                include_outputs_from=["generator"]
            )
            return res["generator"]["replies"][0].text
        except Exception as e:
            st.error(f"Error in paraphraser: {e}")
            return query

class MetaDataFilterPipeline:
    def __init__(self, template):
        self.template = template
        self.pipeline = Pipeline()
        self.pipeline.add_component("materials", GetMaterials())
        self.pipeline.add_component("categories", GetCategories())
        self.pipeline.add_component("prompt_builder", PromptBuilder(
            template=self.template,
            required_variables=["input", "materials", "categories"],
        ))
        self.pipeline.add_component("generator", GroqGenerator())
        self.pipeline.connect("materials.materials", "prompt_builder.materials")
        self.pipeline.connect("categories.categories", "prompt_builder.categories")
        self.pipeline.connect("prompt_builder", "generator")

    def run(self, query: str):
        try:
            res = self.pipeline.run({"prompt_builder": {"input": query}})
            return res["generator"]["replies"][0]
        except Exception as e:
            st.error(f"Error in metadata filter: {e}")
            return "{}"

class RetrieveAndGenerateAnswerPipeline:
    def __init__(self, chat_message_store, document_store):
        self.chat_message_store = chat_message_store
        self.document_store = document_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=8))
        self.pipeline.add_component("prompt_builder", ChatPromptBuilder(
            variables=["query", "documents"],
            required_variables=["query", "documents"]
        ))
        self.pipeline.add_component("generator", GroqChatGenerator())
        
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str, filter: dict = {}):
        messages = [
            ChatMessage.from_system("You are a helpful shop assistant that provides excellent product recommendations."),
            ChatMessage.from_user("""
                Generate a comprehensive list of products that best match the user's query.

                **Query:** {{query}}

                **Product Recommendations:**
                {% if documents|length > 0 %}
                {% for product in documents %}
                **{{loop.index}}. {{ product.meta.title }}**
                💰 **Price:** Rp {{ "{:,}".format(product.meta.price) }}
                🧵 **Material:** {{ product.meta.material }}
                📁 **Category:** {{ product.meta.category }}
                🏷️ **Brand:** {{ product.meta.brand }}
                ⭐ **Why recommended:** {{ product.content[:100] }}...

                {% endfor %}
                {% else %}
                I apologize, but I couldn't find any products matching your criteria. Please try with different specifications or contact our support team for assistance.
                {% endif %}

                Provide helpful guidance and suggest alternatives if needed.
                """)
        ]
        
        try:
            normalized_filter = self.normalize_filter_format(filter)
            
            res = self.pipeline.run({
                "embedder": {"text": query},
                "retriever": {"filters": normalized_filter},
                "prompt_builder": {"query": query, "template": messages}
            }, include_outputs_from=["generator"])
            
            return res["generator"]["replies"][0].text
        except Exception as e:
            st.error(f"Error in product retrieval: {e}")
            return "I apologize, but I'm having trouble retrieving products right now. Please try again."

    def normalize_filter_format(self, filter_dict):
        """Convert simple filter format to Haystack filter format"""
        if not filter_dict:
            return {}
        
        conditions = []
        
        if "category" in filter_dict and filter_dict["category"]:
            conditions.append({
                "field": "category",
                "operator": "in", 
                "value": filter_dict["category"]
            })
        
        if "material" in filter_dict and filter_dict["material"]:
            conditions.append({
                "field": "material",
                "operator": "in",
                "value": filter_dict["material"]
            })
        
        if "price" in filter_dict and isinstance(filter_dict["price"], dict):
            price_filter = filter_dict["price"]
            if "$gte" in price_filter:
                conditions.append({
                    "field": "price",
                    "operator": ">=",
                    "value": price_filter["$gte"]
                })
            if "$lte" in price_filter:
                conditions.append({
                    "field": "price", 
                    "operator": "<=",
                    "value": price_filter["$lte"]
                })
        
        if len(conditions) == 0:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {
                "operator": "AND",
                "conditions": conditions
            }

class CommonInfoPipeline:
    def __init__(self, document_store):
        self.document_store = document_store
        self.pipeline = Pipeline()
        self.pipeline.add_component("embedder", SentenceTransformersTextEmbedder())
        self.pipeline.add_component("retriever", MongoDBAtlasEmbeddingRetriever(document_store=document_store, top_k=5))
        self.pipeline.add_component("prompt_builder", ChatPromptBuilder(
            variables=["query", "documents"],
            required_variables=["query", "documents"]
        ))
        self.pipeline.add_component("generator", GroqChatGenerator())
        
        self.pipeline.connect("embedder", "retriever")
        self.pipeline.connect("retriever", "prompt_builder.documents")
        self.pipeline.connect("prompt_builder.prompt", "generator.messages")

    def run(self, query: str):
        messages = [
            ChatMessage.from_system("You are a helpful customer service assistant for SmartShopper."),
            ChatMessage.from_user("""
                Based on the retrieved information, provide a clear and helpful answer to the user's question.

                **Retrieved Information:**
                {% for doc in documents %}
                **{{ doc.meta.title }}:**
                {{ doc.content }}
                ---
                {% endfor %}

                **User Question:** {{query}}

                Please provide a comprehensive, friendly, and professional answer with clear formatting.
                """)
        ]
        
        try:
            res = self.pipeline.run({
                "embedder": {"text": query},
                "prompt_builder": {"query": query, "template": messages}
            }, include_outputs_from=["generator"])
            
            return res["generator"]["replies"][0].text
        except Exception as e:
            st.error(f"Error in info retrieval: {e}")
            return "I apologize, but I'm having trouble accessing information right now. Please contact our support team."

# Tool Functions
def retrieve_and_generate(query: Annotated[str, "User query for product recommendations"], 
                         paraphraser, metadata_filter, rag_pipeline):
    """Tool for product recommendations with enhanced error handling"""
    try:
        with st.spinner("🔍 Searching for products..."):
            paraphrased_query = paraphraser.run(query)
            filter_result = metadata_filter.run(paraphrased_query)
            
            filter_dict = {}
            try:
                json_match = re.search(r'```json\n(.*?)\n```', filter_result, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    filter_dict = json.loads(json_str)
            except Exception as e:
                filter_dict = {}
            
            result = rag_pipeline.run(paraphrased_query, filter_dict)
            return result
            
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return "I apologize, but I'm having trouble generating recommendations right now. Please try again."

def get_common_information(query: Annotated[str, "User query about shopping information"], 
                          common_info_pipeline):
    """Tool for common shopping information"""
    try:
        with st.spinner("📋 Retrieving information..."):
            result = common_info_pipeline.run(query)
            return result
    except Exception as e:
        st.error(f"Error retrieving information: {e}")
        return "I apologize, but I'm having trouble accessing information right now. Please try again."

def response_handler(query):
    """Enhanced response handler with better error handling"""
    try:
        st.session_state.chat_message_writer.run([ChatMessage.from_user(query)])
        history_messages = list(st.session_state.chat_message_store.messages)
        response = st.session_state.agent.run(messages=history_messages + [ChatMessage.from_user(query)])
        response_text = response["messages"][-1].text
        st.session_state.chat_message_writer.run([ChatMessage.from_assistant(response_text)])
        return response_text
        
    except Exception as e:
        st.error(f"Error processing request: {e}")
        return "I apologize, but I encountered an error while processing your request. Please try again or contact support."

@st.cache_resource
def initialize_document_stores():
    """Initialize document stores with caching"""
    try:
        products_store = MongoDBAtlasDocumentStore(
            database_name="smartshopper_store",
            collection_name="products",
            vector_search_index="vector_index",
            full_text_search_index="search_index",
        )
        
        common_info_store = MongoDBAtlasDocumentStore(
            database_name="smartshopper_store",
            collection_name="common_info",
            vector_search_index="common_info_vector_index",
            full_text_search_index="common_info_search_index",
        )
        
        return products_store, common_info_store
    except Exception as e:
        st.error(f"Error initializing document stores: {e}")
        return None, None

def main():
    # Page configuration
    st.set_page_config(
        page_title="SmartShopper Assistant",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Header with improved styling
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🛍️ SmartShopper Assistant</div>
        <div class="header-subtitle">Your Intelligent Shopping Companion</div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize document stores
    if 'document_stores_initialized' not in st.session_state:
        with st.spinner("🔧 Initializing system components..."):
            products_store, common_info_store = initialize_document_stores()
            if products_store and common_info_store:
                st.session_state.products_document_store = products_store
                st.session_state.common_info_document_store = common_info_store
                st.session_state.document_stores_initialized = True
                
                # Success message with proper styling
                st.markdown("""
                <div class="status-success">
                    ✅ System initialized successfully!
                </div>
                """, unsafe_allow_html=True)
            else:
                # Error message with proper styling
                st.markdown("""
                <div class="status-error">
                    ❌ Failed to initialize system. Please check your configuration.
                </div>
                """, unsafe_allow_html=True)
                return

    # Initialize session state components
    if 'chat_message_store' not in st.session_state:
        st.session_state.chat_message_store = InMemoryChatMessageStore()
    
    if 'chat_message_writer' not in st.session_state:
        st.session_state.chat_message_writer = ChatMessageWriter(st.session_state.chat_message_store)
    
    if 'pipelines_initialized' not in st.session_state:
        with st.spinner("🔧 Setting up AI pipelines..."):
            try:
                # Initialize pipelines
                st.session_state.paraphraser_pipeline = ParaphraserPipeline(st.session_state.chat_message_store)
                st.session_state.metadata_filter_pipeline = MetaDataFilterPipeline(METADATA_FILTER_TEMPLATE)