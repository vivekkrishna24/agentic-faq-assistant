import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter
from typing import List
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class FAQTextSplitter(TextSplitter):
    """Custom text splitter for FAQ documents with Q&A pairs separated by empty lines."""
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into Q&A pairs.
        
        Args:
            text (str): The text to split
            
        Returns:
            List[str]: List of Q&A pairs
        """
        # Split by double newlines to separate Q&A pairs
        pairs = text.strip().split('\n\n')
        
        # Filter out empty pairs and clean up whitespace
        cleaned_pairs = []
        for pair in pairs:
            if pair.strip():
                # Clean up any extra whitespace within the pair
                cleaned_pair = ' '.join(line.strip() for line in pair.split('\n') if line.strip())
                cleaned_pairs.append(cleaned_pair)
                
        return cleaned_pairs

def ingest_documents(
    document_path: str,
    collection_name: str = "rag_faq",
    persist_directory: str = "rag_faq",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Ingest documents into the vector store.
    
    Args:
        document_path (str): Path to the document file
        collection_name (str): Name of the collection in the vector store
        persist_directory (str): Directory to persist the vector store
        chunk_size (int): Size of each text chunk
        chunk_overlap (int): Overlap between chunks
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    # Initialize text splitter
    text_splitter = FAQTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Read the document
    with open(document_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split the text into chunks
    chunks = text_splitter.split_text(text)
    
    # Create a new Chroma vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Persist the vector store to disk
    vectorstore.persist()
    
    print(f"Ingested {len(chunks)} chunks into the vector store at {persist_directory}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ingest documents into the vector store.')
    parser.add_argument('--document', type=str, required=True, help='Path to the document file')
    parser.add_argument('--collection', type=str, default="rag_faq", help='Name of the collection')
    parser.add_argument('--persist-dir', type=str, default="rag_faq", help='Directory to persist the vector store')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Size of each text chunk')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Overlap between chunks')
    
    args = parser.parse_args()
    
    ingest_documents(
        document_path=args.document,
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
