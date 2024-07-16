# backend.py

import torch
import sys
import threading
import logging
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log system information
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name()}")

class StreamingCallback(BaseCallbackHandler):
    def __init__(self, stream_callback: Callable[[str], None]):
        self.stream_callback = stream_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        try:
            self.stream_callback(token)
        except Exception as e:
            logger.error(f"Error in StreamingCallback.on_llm_new_token: {str(e)}")

class LLMBackend:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_model: HuggingFaceEmbeddings = None
        self.vectorstore: Chroma = None
        self.llm: LlamaCpp = None
        self.rag_pipeline: RetrievalQA = None
        self.pdf_thread: threading.Thread = None
        self.is_pdf_processed: bool = False
        self.llm_models: Dict[str, str] = {
            "Llama 2 - 7B": "E:\\PDF-LLM\\models\\llama-2-7b-chat.Q4_K_M.gguf",
            # "Llama 2 - 13B": "E:\\PDF-LLM\\models\\llama-2-13b-chat.Q4_K_M.gguf",
            # "Llama 3 - 11.5B": "E:\\PDF-LLM\\models\\Llama-3-11.5B-V2-Q4_K_M.gguf",
        }
        self.current_llm: str = "Llama 2 - 7B"  # Default model
        self.stream_callback: Callable[[str], None] = None

    def initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            model_name = "sentence-transformers/all-mpnet-base-v2"
            self.embed_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': self.device},
                encode_kwargs={'device': self.device, 'batch_size': 32}
            )
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

    def set_current_llm(self, llm_name: str, callback: Callable[[str], None]) -> Tuple[bool, str]:
        """Set the current LLM model and load it if a PDF has been processed."""
        if llm_name in self.llm_models:
            self.current_llm = llm_name
            if self.is_pdf_processed:
                return self.load_current_model(callback)
            return True, "Model selected, but not loaded. Please load a PDF first."
        return False, f"Unknown LLM: {llm_name}"

    def load_current_model(self, callback: Callable[[str], None]) -> Tuple[bool, str]:
        """Load the current LLM model."""
        try:
            n_gpu_layers = 32 if torch.cuda.is_available() else 0
            n_batch = 512
            callback_manager = CallbackManager([StreamingCallback(self.stream_callback)])

            self.llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

            if self.is_pdf_processed and self.vectorstore:
                self.setup_rag_pipeline(callback)
            
            logger.info(f"{self.current_llm} loaded successfully")
            return True, f"{self.current_llm} loaded successfully"
        except Exception as e:
            logger.error(f"Error loading {self.current_llm}: {str(e)}")
            return False, f"Error loading {self.current_llm}: {str(e)}"

    def load_pdf(self, file_path: str, progress_callback: Callable[[int, str], None], completion_callback: Callable[[bool, str], None]) -> None:
        """Load and process a PDF file."""
        def process_pdf() -> None:
            try:
                progress_callback(0, "Loading PDF...")
                loader = PyPDFLoader(file_path)
                data = loader.load()
                
                progress_callback(33, "Splitting text...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                all_splits = text_splitter.split_documents(data)
                
                progress_callback(66, "Creating vector store...")
                self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.embed_model)
                
                self.is_pdf_processed = True
                logger.info("PDF processed successfully")
                completion_callback(True, "PDF processed successfully!")
            except Exception as e:
                logger.error(f"Error processing PDF: {str(e)}")
                self.is_pdf_processed = False
                completion_callback(False, f"Error processing PDF: {str(e)}")

        self.pdf_thread = threading.Thread(target=process_pdf)
        self.pdf_thread.start()

    def setup_rag_pipeline(self, stream_callback: Callable[[str], None]) -> Tuple[bool, str]:
        """Set up the Retrieval-Augmented Generation (RAG) pipeline."""
        try:
            if not self.is_pdf_processed or self.vectorstore is None:
                raise ValueError("PDF is not processed or vectorstore is not initialized. Please load a PDF first.")

            self.stream_callback = stream_callback

            if not self.llm:
                success, message = self.load_current_model(stream_callback)
                if not success:
                    return False, message

            template = """
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            Use three sentences maximum and keep the answer concise.
            {context}
            Question: {question}
            Helpful Answer:"""
            QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

            self.rag_pipeline = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type='stuff',
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
            logger.info("RAG pipeline set up successfully")
            return True, "RAG pipeline set up successfully"
        except Exception as e:
            logger.error(f"Error setting up RAG pipeline: {str(e)}")
            return False, f"Error setting up RAG pipeline: {str(e)}"

    def ask_question(self, question: str, callback: Callable[[str], None]) -> None:
        """Process a question using the RAG pipeline."""
        def process_question() -> None:
            try:
                if self.rag_pipeline:
                    result = self.rag_pipeline(question)
                    callback(result.get('result', "No result found."))
                else:
                    callback("Please load a PDF and set up the pipeline first.")
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                callback(f"An error occurred while processing your question: {str(e)}")

        threading.Thread(target=process_question).start()

# Example usage
if __name__ == "__main__":
    backend = LLMBackend()
    backend.initialize()
    print("Backend initialized. Ready for use in the main application.")