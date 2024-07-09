# frontend.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from backend import LLMBackend
import customtkinter as ctk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class LLMUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.backend = LLMBackend()
        self.backend.initialize()

        self.title("PDF Question Answering")
        self.geometry("1000x600")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.create_widgets()
        self.question_entry.bind("<Return>", self.ask_question)

    def create_widgets(self):
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="PDF Q&A", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.upload_button = ctk.CTkButton(self.sidebar_frame, text="Upload PDF", command=self.upload_pdf)
        self.upload_button.grid(row=1, column=0, padx=20, pady=10)
        
        self.pdf_status_label = ctk.CTkLabel(self.sidebar_frame, text="No PDF uploaded")
        self.pdf_status_label.grid(row=2, column=0, padx=20, pady=(10, 0))
        
        # Create a spacer frame to push the following widgets to the bottom
        spacer = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        spacer.grid(row=3, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(3, weight=1)
        
        self.llm_label = ctk.CTkLabel(self.sidebar_frame, text="Select LLM:")
        self.llm_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="sw")
        
        self.llm_var = tk.StringVar(value="Llama 2")
        self.llm_dropdown = ctk.CTkOptionMenu(self.sidebar_frame, values=["Llama 2"], variable=self.llm_var)
        self.llm_dropdown.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="sw")

        # Main area
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # Chat area
        self.chat_frame = ctk.CTkScrollableFrame(self.main_frame)
        self.chat_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))
        self.chat_frame.grid_columnconfigure(0, weight=1)

        # Input area
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=1, column=0, sticky="sew", padx=10, pady=10)
        self.input_frame.grid_columnconfigure(0, weight=1)

        self.question_entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your question here...")
        self.question_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.question_entry.bind("<Return>", self.ask_question)

        self.ask_button = ctk.CTkButton(self.input_frame, text="Ask", width=100, command=self.ask_question)
        self.ask_button.grid(row=0, column=1)

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.pdf_status_label.configure(text="Processing PDF...")
            self.backend.load_pdf(file_path, self.update_progress, self.pdf_processing_complete)

    def update_progress(self, value, message):
        self.pdf_status_label.configure(text=message)

    def pdf_processing_complete(self, success, message):
        if success:
            self.pdf_status_label.configure(text=message)
            self.setup_rag_pipeline()
        else:
            self.pdf_status_label.configure(text=message)
            messagebox.showerror("Error", message)

    def setup_rag_pipeline(self):
        self.pdf_status_label.configure(text="Setting up RAG pipeline...")
        success, message = self.backend.setup_rag_pipeline(self.stream_callback)
        if success:
            self.pdf_status_label.configure(text=message)
        else:
            self.pdf_status_label.configure(text=message)
            messagebox.showerror("Error", message)

    def stream_callback(self, token):
        if not hasattr(self, 'current_message'):
            self.current_message = ctk.CTkTextbox(self.chat_frame, height=100, wrap="word", state="normal")
            self.current_message.grid(sticky="ew", padx=5, pady=5)
        self.current_message.configure(state="normal")
        self.current_message.insert(tk.END, token)
        self.current_message.see(tk.END)
        self.current_message.configure(state="disabled")

    def ask_question(self, event=None):
        if event:
            event.preventDefault()
        question = self.question_entry.get()
        if question:
            if not self.backend.is_pdf_processed:
                messagebox.showwarning("Warning", "Please upload and process a PDF first.")
                return

            self.ask_button.configure(state="disabled")
            
            question_label = ctk.CTkTextbox(self.chat_frame, height=30, wrap="word")
            question_label.insert("1.0", f"Q: {question}")
            question_label.configure(state="disabled")
            question_label.grid(sticky="ew", padx=5, pady=5)

            self.current_message = ctk.CTkTextbox(self.chat_frame, height=100, wrap="word")
            self.current_message.grid(sticky="ew", padx=5, pady=5)
            
            self.backend.ask_question(question, self.process_answer)
        else:
            messagebox.showwarning("Warning", "Please enter a question.")
        
        self.question_entry.delete(0, 'end')
    
    def process_answer(self, answer):
        self.current_message.configure(state="normal")
        self.current_message.delete("1.0", tk.END)
        self.current_message.insert(tk.END, answer)
        self.current_message.configure(state="disabled")
        self.ask_button.configure(state="normal")

if __name__ == "__main__":
    app = LLMUI()
    app.mainloop()