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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
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
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif !important;
    }
    
    .main {
        padding: 1rem 2rem;
        background-color: var(--light-bg);
    }
    
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
    
    .status-warning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404 !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid var(--warning-color);
        margin: 1rem 0;
        font-weight: 600;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 193, 7, 0.2);
    }
    
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
    </style>
    """, unsafe_allow_html=True)

# Enhanced Components with better error handling
@component
class GroqChatGenerator:
    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: str = None):
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        self.model = model
    
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], tools: List = None):
        if not messages:
            return {"replies": [ChatMessage.from_assistant("Hello! How can I help you today?")]}
            
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
            st.error(f"Groq API Error: {e}")
            return {
                "replies": [
                    ChatMessage.from_assistant("I'm experiencing technical difficulties. Please try again.")
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
            return {"replies": ["{}"]}
            
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
            st.error(f"Groq API Error: {e}")
            return {"replies": ["{}"]}

class MongoDBConnectionManager:
    """Singleton connection manager with proper timeout and retry logic"""
    _instance = None
    _client = None
    
    def __new__(cls, connection_string: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, connection_string: str):
        if self._client is None:
            try:
                # Connection with proper timeout settings
                self._client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000,  # 5 seconds timeout
                    connectTimeoutMS=5000,
                    socketTimeoutMS=5000,
                    maxPoolSize=10,
                    retryWrites=True
                )
                # Test connection
                self._client.admin.command('ping')
            except Exception as e:
                st.error(f"MongoDB connection failed: {e}")
                self._client = None
    
    def get_client(self):
        return self._client
    
    def get_database(self, db_name: str):
        if self._client:
            return self._client[db_name]
        return None

@component  
class GetMaterials:
    def __init__(self):
        self.connection_string = os.environ.get('MONGO_CONNECTION_STRING')
        
    @component.output_types(materials=List[str])
    def run(self):
        if not self.connection_string:
            st.error("MongoDB connection string not found")
            return {"materials": ["Cotton", "Polyester", "Wool", "Leather"]}  # Fallback
        
        try:
            manager = MongoDBConnectionManager(self.connection_string)
            db = manager.get_database("smartshopper_store")
            if db is None:
                raise Exception("Database connection failed")
                
            materials = [doc['name'] for doc in db.materials.find().limit(100)]
            return {"materials": materials if materials else ["Cotton", "Polyester", "Wool", "Leather"]}
        except Exception as e:
            st.warning(f"Using fallback materials due to: {e}")
            return {"materials": ["Cotton", "Polyester", "Wool", "Leather", "Silk", "Denim"]}

@component
class GetCategories:
    def __init__(self):
        self.connection_string = os.environ.get('MONGO_CONNECTION_STRING')
        
    @component.output_types(categories=List[str])
    def run(self):
        if not self.connection_string:
            st.error("MongoDB connection string not found")
            return {"categories": ["Clothing", "Electronics", "Home", "Sports"]}  # Fallback
        
        try:
            manager = MongoDBConnectionManager(self.connection_string)
            db = manager.get_database("smartshopper_store")
            if db is None:
                raise Exception("Database connection failed")
                
            categories = [doc['name'] for doc in db.categories.find().limit(100)]
            return {"categories": categories if categories else ["Clothing", "Electronics", "Home", "Sports"]}
        except Exception as e:
            st.warning(f"Using fallback categories due to: {e}")
            return {"categories": ["Clothing", "Electronics", "Home", "Sports", "Beauty", "Books"]}

# Enhanced Templates
METADATA_FILTER_TEMPLATE = """
You are an expert assistant that helps create metadata filters for product searches. 
Based on the user input, create a JSON filter object that can be used to filter products.

Available materials: {{materials}}
Available categories: {{categories}}

The filter should follow this structure:
```json
{
    "material": ["material1", "material2"],
    "category": ["category1", "category2"],
    "price": {"$gte": min_price, "$lte": max_price}
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
            st.warning(f"Paraphraser error, using original query: {e}")
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
            st.warning(f"Filter generation error, using empty filter: {e}")
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
                📂 **Category:** {{ product.meta.category }}
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
            st.error(f"Product retrieval error: {e}")
            return "I apologize, but I'm having trouble retrieving products right now. Please try again or contact support."

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

                Please provide a comprehensive, friendly, and professional answer.
                """)
        ]
        
        try:
            res = self.pipeline.run({
                "embedder": {"text": query},
                "prompt_builder": {"query": query, "template": messages}
            }, include_outputs_from=["generator"])
            
            return res["generator"]["replies"][0].text
        except Exception as e:
            st.error(f"Info retrieval error: {e}")
            return "I apologize, but I'm having trouble accessing information right now. Please contact our support team."

# Tool Functions with better error handling
def retrieve_and_generate(query: Annotated[str, "User query for product recommendations"], 
                         paraphraser, metadata_filter, rag_pipeline):
    """Tool for product recommendations"""
    try:
        paraphrased_query = paraphraser.run(query)
        filter_result = metadata_filter.run(paraphrased_query)
        
        filter_dict = {}
        try:
            json_match = re.search(r'```json\n(.*?)\n```', filter_result, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                filter_dict = json.loads(json_str)
        except:
            filter_dict = {}
        
        result = rag_pipeline.run(paraphrased_query, filter_dict)
        return result
        
    except Exception as e:
        return f"I apologize, but I'm having trouble generating recommendations right now. Error: {e}"

def get_common_information(query: Annotated[str, "User query about shopping information"], 
                          common_info_pipeline):
    """Tool for common shopping information"""
    try:
        result = common_info_pipeline.run(query)
        return result
    except Exception as e:
        return f"I apologize, but I'm having trouble accessing information right now. Error: {e}"

def response_handler(query):
    """Enhanced response handler"""
    try:
        if not hasattr(st.session_state, 'agent'):
            return "System not properly initialized. Please refresh the page."
            
        st.session_state.chat_message_writer.run([ChatMessage.from_user(query)])
        history_messages = list(st.session_state.chat_message_store.messages)
        response = st.session_state.agent.run(messages=history_messages + [ChatMessage.from_user(query)])
        response_text = response["messages"][-1].text
        st.session_state.chat_message_writer.run([ChatMessage.from_assistant(response_text)])
        return response_text
        
    except Exception as e:
        st.error(f"Response handler error: {e}")
        return "I apologize, but I encountered an error. Please try again."

def test_connections():
    """Test all required connections"""
    errors = []
    
    # Test Groq API
    try:
        if not os.environ.get("GROQ_API_KEY"):
            errors.append("GROQ_API_KEY not found in environment variables")
        else:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=10
            )
    except Exception as e:
        errors.append(f"Groq API connection failed: {e}")
    
    # Test MongoDB connection
    try:
        if not os.environ.get("MONGO_CONNECTION_STRING"):
            errors.append("MONGO_CONNECTION_STRING not found in environment variables")
        else:
            manager = MongoDBConnectionManager(os.environ.get("MONGO_CONNECTION_STRING"))
            if manager.get_client() is None:
                errors.append("MongoDB connection failed")
    except Exception as e:
        errors.append(f"MongoDB connection error: {e}")
    
    return errors

@st.cache_resource
def initialize_document_stores():
    """Initialize document stores with better error handling"""
    connection_string = os.environ.get('MONGO_CONNECTION_STRING')
    if not connection_string:
        st.error("MongoDB connection string not found in environment variables")
        return None, None
    
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
        st.error(f"Document store initialization failed: {e}")
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
    
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🛍️ SmartShopper Assistant</div>
        <div class="header-subtitle">Your Intelligent Shopping Companion</div>
    </div>
    """, unsafe_allow_html=True)

    # Test connections first
    if 'connection_tested' not in st.session_state:
        with st.spinner("🔍 Testing system connections..."):
            connection_errors = test_connections()
            if connection_errors:
                st.markdown("""
                <div class="status-error">
                    ❌ <strong>Connection Issues Detected:</strong><br>
                    """ + "<br>".join(f"• {error}" for error in connection_errors) + """
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="status-warning">
                    ⚠️ <strong>To fix these issues:</strong><br>
                    • Check your .env file contains GROQ_API_KEY and MONGO_CONNECTION_STRING<br>
                    • Verify your MongoDB Atlas cluster is running and accessible<br>
                    • Check your internet connection<br>
                    • Ensure MongoDB Atlas IP whitelist includes your current IP
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("🔄 Retry Connection"):
                    del st.session_state.connection_tested
                    st.rerun()
                return
            else:
                st.session_state.connection_tested = True
                st.markdown("""
                <div class="status-success">
                    ✅ All connections successful!
                </div>
                """, unsafe_allow_html=True)

    # Initialize document stores
    if 'document_stores_initialized' not in st.session_state:
        with st.spinner("📊 Initializing document stores..."):
            products_store, common_info_store = initialize_document_stores()
            if products_store and common_info_store:
                st.session_state.products_document_store = products_store
                st.session_state.common_info_document_store = common_info_store
                st.session_state.document_stores_initialized = True
            else:
                st.markdown("""
                <div class="status-error">
                    ❌ Failed to initialize document stores. Please check your MongoDB configuration.
                </div>
                """, unsafe_allow_html=True)
                return

    # Initialize chat components
    if 'chat_message_store' not in st.session_state:
        st.session_state.chat_message_store = InMemoryChatMessageStore()
    
    if 'chat_message_writer' not in st.session_state:
        st.session_state.chat_message_writer = ChatMessageWriter(st.session_state.chat_message_store)
    
    # Initialize pipelines
    if 'pipelines_initialized' not in st.session_state:
        with st.spinner("🔧 Setting up AI pipelines..."):
            try:
                st.session_state.paraphraser_pipeline = ParaphraserPipeline(st.session_state.chat_message_store)
                st.session_state.metadata_filter_pipeline = MetaDataFilterPipeline(METADATA_FILTER_TEMPLATE)
                st.session_state.rag_pipeline = RetrieveAndGenerateAnswerPipeline(
                    st.session_state.chat_message_store, 
                    st.session_state.products_document_store
                )
                st.session_state.common_info_pipeline = CommonInfoPipeline(st.session_state.common_info_document_store)

                # Initialize agent with tools
                retrieval_tool = Tool(
                    name="retrieve_and_generate",
                    description="Use this tool to search and recommend products based on user queries about shopping, products, or recommendations.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    },
                    function=partial(
                        retrieve_and_generate,
                        paraphraser=st.session_state.paraphraser_pipeline,
                        metadata_filter=st.session_state.metadata_filter_pipeline,
                        rag_pipeline=st.session_state.rag_pipeline
                    )
                )

                common_info_tool = Tool(
                    name="get_common_information", 
                    description="Use this tool for general shopping information, policies, FAQs, and common questions that don't require specific product searches.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    },
                    function=partial(
                        get_common_information,
                        common_info_pipeline=st.session_state.common_info_pipeline
                    )
                )

                st.session_state.agent = Agent(
                    chat_generator=GroqChatGenerator(),
                    tools=[retrieval_tool, common_info_tool]
                )

                st.session_state.pipelines_initialized = True
                
                st.markdown("""
                <div class="status-success">
                    ✅ AI pipelines initialized successfully!
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="status-error">
                    ❌ Failed to initialize AI pipelines: {str(e)}
                </div>
                """, unsafe_allow_html=True)
                return

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h3>💡 Features</h3>
            <p>Explore what I can help you with</p>
        </div>
        """, unsafe_allow_html=True)
        
        features = [
            "🔍 Product Search & Recommendations",
            "💰 Price Comparisons", 
            "📊 Category Filtering",
            "🏷️ Brand Information",
            "❓ Shopping Assistance"
        ]
        
        for feature in features:
            st.markdown(f'<div class="feature-box">{feature}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sample questions
        st.markdown("### 🔍 Try these examples:")
        
        sample_questions = [
            "Show me winter jackets under 500k",
            "What are the best running shoes?", 
            "I need a laptop for gaming",
            "Cotton shirts for office wear",
            "What's your return policy?",
            "How do I track my order?",
            "Do you offer warranties?",
            "What payment methods do you accept?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}", use_container_width=True):
                st.session_state.sample_query = question

        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            if hasattr(st.session_state, 'chat_message_store'):
                st.session_state.chat_message_store.messages.clear()
                st.rerun()

    # Main chat interface
    st.markdown("### 💬 Chat with SmartShopper Assistant")
    
    # Display chat messages
    if hasattr(st.session_state, 'chat_message_store'):
        for message in st.session_state.chat_message_store.messages:
            role = "user" if message.role.value == "user" else "assistant"
            with st.chat_message(role):
                st.markdown(message.text)

    # Handle sample query
    if hasattr(st.session_state, 'sample_query'):
        query = st.session_state.sample_query
        del st.session_state.sample_query
        
        with st.chat_message("user"):
            st.markdown(query)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                if hasattr(st.session_state, 'agent'):
                    response = response_handler(query)
                    st.markdown(response)
                else:
                    st.error("System not properly initialized. Please refresh the page.")

    # Chat input
    if prompt := st.chat_input("What can I help you find today? 🛍️"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                if hasattr(st.session_state, 'agent'):
                    response = response_handler(prompt)
                    st.markdown(response)
                else:
                    st.error("System not properly initialized. Please refresh the page.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: var(--text-secondary);">
        <p>🛍️ <strong>SmartShopper Assistant</strong> - Powered by AI | 
        Need help? Contact our support team!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()