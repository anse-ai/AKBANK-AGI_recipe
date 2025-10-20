import streamlit as st
import os
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- (This function will be cached, running only once) ---
@st.cache_resource
def setup_rag_pipeline():
    """
    Loads data, creates embeddings, builds the vector store,
    and sets up the RAG chain.
    """
    # --- Part 1: Load Data ---
    #
    st.write("Loading dataset...") 
    dataset = load_dataset("m3hrdadfi/recipe_nlg_lite", split="train")
    
    documents = []
    for recipe in dataset:
        ingredients = ", ".join(recipe.get("ingredients", []))
        steps = " ".join(recipe.get("steps", []))
        page_content = f"Recipe Name: {recipe.get('name', '')}\n" \
                       f"Ingredients: {ingredients}\n" \
                       f"Instructions: {steps}"
        metadata = {"source_link": recipe.get("link", "No source available")}
        documents.append(Document(page_content=page_content, metadata=metadata))

    # --- Part 2: Indexing (Vector Store) ---
    st.write("Initializing embedding model...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #
    
    st.write("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(documents, embeddings) #
    retriever = vector_store.as_retriever()

    # --- Part 3: RAG Chain ---
    st.write("Setting up RAG chain...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") #
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert cooking assistant. Answer the user's question based *only* on the following context (recipes):
    <context>{context}</context>
    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt) #
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    st.write("RAG chain is ready!")
    return retrieval_chain

# --- Page Configuration ---
st.set_page_config(page_title="🍳 Recipe Chatbot", page_icon="🍳")
st.title("🍳 Recipe Chatbot")
st.markdown("Ask me what you can cook with the ingredients you have!")

# --- API Key Setup ---
# This fulfills part of the requirement for a "running guide"
# by handling API keys securely.
try:
    # Try to get the key from Streamlit's secrets
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except:
    # Handle the case where the secret isn't set
    st.error("GOOGLE_API_KEY not found in Streamlit secrets. Please add it to .streamlit/secrets.toml")
    st.stop() # Stop the app if the key is missing

# --- Load the RAG chain (from cache) ---
try:
    rag_chain = setup_rag_pipeline()
except Exception as e:
    st.error(f"Error setting up RAG pipeline: {e}")
    st.stop()


# --- Chat Interface ---
#

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What do you want to cook?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Call the RAG chain
            response = rag_chain.invoke({"input": prompt})
            answer = response['answer']
            
            # --- Bonus: Add sources from the retrieved context ---
            sources = [doc.metadata.get('source_link', 'No source') for doc in response.get('context', [])]
            unique_sources = list(set(sources)) # Get unique links
            
            if unique_sources:
                 answer += f"\n\n**Sources:**\n" + "\n".join(f"- {src}" for src in unique_sources if src != 'No source')

            st.markdown(answer)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
