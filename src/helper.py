from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_pdf_file(data_path):
    """Load PDF files from a directory."""
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_embeddings():
    """Initialize and return the embedding model."""
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )

def initialize_vectorstore(index_name, embeddings):
    """Initialize Pinecone vector store with existing index."""
    return PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

def setup_llm():
    """Initialize the Ollama LLM."""
    return ChatOllama(
        model="llama3.2:1b",
        base_url="http://localhost:11434",
        temperature=0
    )

def create_rag_chain(retriever, llm_engine, prompt_template):
    """Create the RAG chain for question answering."""
    question_answer_chain = create_stuff_documents_chain(
        llm_engine,
        prompt_template
    )
    return create_retrieval_chain(retriever, question_answer_chain)