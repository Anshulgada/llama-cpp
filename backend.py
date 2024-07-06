# backend.py

import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")



import threading
import queue
# import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.callbacks.base import BaseCallbackHandler

class StreamingCallback(BaseCallbackHandler):
    def __init__(self, stream_callback):
        self.stream_callback = stream_callback

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.stream_callback(token)

class LLMBackend:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_model = None
        self.vectorstore = None
        self.llm = None
        self.rag_pipeline = None
        self.pdf_queue = queue.Queue()
        self.pdf_thread = None
        self.is_pdf_processed = False

    def initialize(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'device': self.device, 'batch_size': 32}
        )

    def load_pdf(self, file_path, progress_callback, completion_callback):
        def process_pdf():
            try:
                progress_callback(0, "Loading PDF...")
                loader = PyPDFLoader(file_path)
                data = loader.load()
                progress_callback(33, "Splitting text...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
                all_splits = text_splitter.split_documents(data)
                progress_callback(66, "Creating vector store...")
                self.vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.embed_model)
                progress_callback(100, "PDF processed successfully!")
                self.is_pdf_processed = True
                completion_callback(True, "PDF processed successfully!")
            except Exception as e:
                self.is_pdf_processed = False
                completion_callback(False, f"Error processing PDF: {str(e)}")

        self.pdf_thread = threading.Thread(target=process_pdf)
        self.pdf_thread.start()

    def setup_rag_pipeline(self, stream_callback):
        if not self.is_pdf_processed or self.vectorstore is None:
            raise ValueError("PDF is not processed or vectorstore is not initialized. Please load a PDF first.")

        n_gpu_layers = 32 if torch.cuda.is_available() else 0  # Use GPU layers only if CUDA is available
        n_batch = 512
        callback_manager = CallbackManager([StreamingCallback(stream_callback)])

        self.llm = LlamaCpp(
            model_path="E:\\PDF-LLM\\llama-2-7b-chat.Q4_K_M.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=2048,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=True,
            use_mlock=False,
            use_mmap=True,
            seed=-1,
        )

        template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as long or short as you like, there are no restrictions to the length. Provide answers using information stored in the vector store. If the answer to a question is known by you but not stored in the vector store, do not provide an answer.
        {context}
        Question: {question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        self.rag_pipeline = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

    def ask_question(self, question, callback):
        def process_question():
            if self.rag_pipeline:
                result = self.rag_pipeline(question)
                callback(result.get('result', "No result found."))
            else:
                callback("Please load a PDF and set up the pipeline first.")

        thread = threading.Thread(target=process_question)
        thread.start()

# Check if CUDA is available and set the default tensor type accordingly
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    print("CUDA is not available. Using CPU.")