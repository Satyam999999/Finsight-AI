import streamlit as st
import os
import re
from dotenv import load_dotenv

# --- FIX for Mutex Lock Error on macOS ---
# This MUST be set before any other imports from langchain or transformers.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# -----------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Load environment variables
load_dotenv()

# --- Constants ---
PERSIST_DIRECTORY = "chroma_db_persistent"

# --- Core Application Logic ---

@st.cache_resource(show_spinner="Initializing Vector Store...")
def get_vectorstore():
    """Initializes and returns a persistent Chroma vector store."""
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    # Initialize a persistent ChromaDB instance
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return vectorstore

def process_and_store_documents(uploaded_files, vectorstore):
    """Loads, splits, and stores documents in the persistent vector store."""
    processed_files = []
    with st.spinner("Processing and embedding documents... This may take a while."):
        for uploaded_file in uploaded_files:
            # Create a temporary file to load the document
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()

            # Add source metadata to each document for later filtering
            for doc in docs:
                doc.metadata["source"] = uploaded_file.name
            
            # Setup retrievers and add documents
            parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
            child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
            store = InMemoryStore() # In-memory store for parent docs for this session
            
            retriever = ParentDocumentRetriever(
                vectorstore=vectorstore,
                docstore=store,
                child_splitter=child_splitter,
                parent_splitter=parent_splitter,
            )
            retriever.add_documents(docs)
            processed_files.append(uploaded_file.name)
            
            # Clean up the temporary file
            os.remove(uploaded_file.name)
            
    st.success(f"Successfully processed and indexed: {', '.join(processed_files)}")
    return processed_files

def get_conversational_chain(vectorstore, selected_document, google_api_key):
    """Creates and returns a conversational retrieval chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0.3,
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )
    
    # Set up memory to remember the last 3 conversation turns
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )

    # Base retriever that searches across all documents
    retriever = vectorstore.as_retriever()
    
    # If a specific document is selected, modify the retriever to filter by it
    if selected_document != "All Documents":
        retriever.search_kwargs = {'filter': {'source': selected_document}}

    # Create the conversational chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )
    return chain

# --- Streamlit UI ---

st.set_page_config(page_title="Financial Analyzer V2 🧠", layout="wide")
st.title("Financial Document Analyzer V2: Conversational & Persistent")

# Initialize vector store and session state
vectorstore = get_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

# Sidebar for configuration and document management
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input(
        "Google API Key:", type="password", value=os.getenv("GOOGLE_API_KEY", "")
    )
    
    st.markdown("---")
    st.header("Document Library")
    
    # Get a unique list of documents already in the database
    db_docs = vectorstore.get()
    if db_docs['ids']:
        sources = list(set([m['source'] for m in db_docs['metadatas']]))
        st.write("Indexed Documents:")
        for source in sorted(sources):
            st.markdown(f"- `{source}`")
    else:
        sources = []
        st.info("No documents have been indexed yet.")

    uploaded_files = st.file_uploader(
        "Upload new PDF documents", type="pdf", accept_multiple_files=True
    )

    if st.button("Process New Documents") and uploaded_files:
        process_and_store_documents(uploaded_files, vectorstore)
        # Force a rerun to update the list of indexed docs
        st.rerun()

    st.markdown("---")
    # Dropdown to select a document for conversation
    if sources:
        doc_options = ["All Documents"] + sorted(sources)
        selected_document = st.selectbox(
            "Chat with a specific document:", options=doc_options
        )
    else:
        selected_document = "All Documents"


# Create the conversational chain if it doesn't exist
if google_api_key:
    st.session_state.conversation_chain = get_conversational_chain(
        vectorstore, selected_document, google_api_key
    )

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input
if prompt := st.chat_input(f"Ask a question about '{selected_document}'..."):
    if not google_api_key:
        st.warning("Please enter your Google API Key in the sidebar to start.")
    elif not sources and not uploaded_files:
        st.warning("Please upload and process at least one document to begin.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.conversation_chain({"question": prompt})
                response = result['answer']
                
                # Format the response with source documents
                response_with_sources = f"{response}\n\n---"
                if result.get("source_documents"):
                    response_with_sources += "\n\n**Sources:**\n"
                    for doc in result["source_documents"]:
                        source = doc.metadata.get('source', 'Unknown')
                        page = doc.metadata.get('page', 'N/A')
                        response_with_sources += f"- *{source}, page {page}*\n"
                
                st.markdown(response_with_sources)
                
        st.session_state.messages.append({"role": "assistant", "content": response_with_sources})
