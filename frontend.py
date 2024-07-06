# frontend.py

import tkinter as tk
from tkinter import ttk, filedialog
from backend import LLMBackend

class LLMUI:
    def __init__(self, master):
        self.master = master
        self.backend = LLMBackend()
        self.backend.initialize()

        master.title("LLM Question Answering")
        master.geometry("800x600")
        master.configure(bg='#2b2b2b')

        self.create_widgets()

    def create_widgets(self):
        # File Upload
        upload_frame = tk.Frame(self.master, bg='#2b2b2b')
        upload_frame.pack(pady=10)

        upload_button = tk.Button(upload_frame, text="Upload PDF", command=self.upload_pdf, bg='#3c3f41', fg='white')
        upload_button.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(upload_frame, text="No file selected", bg='#2b2b2b', fg='white')
        self.file_label.pack(side=tk.LEFT)

        # PDF Status Label
        self.pdf_status_label = tk.Label(self.master, text="No PDF uploaded", bg='#2b2b2b', fg='white')
        self.pdf_status_label.pack(pady=5)

        # Progress Bar
        self.progress_frame = tk.Frame(self.master, bg='#2b2b2b')
        self.progress_frame.pack(pady=10, padx=10, fill=tk.X)

        self.progress_bar = ttk.Progressbar(self.progress_frame, length=300, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.progress_label = tk.Label(self.progress_frame, text="", bg='#2b2b2b', fg='white')
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # LLM Selection
        llm_frame = tk.Frame(self.master, bg='#2b2b2b')
        llm_frame.pack(pady=10)

        tk.Label(llm_frame, text="Select LLM:", bg='#2b2b2b', fg='white').pack(side=tk.LEFT, padx=5)
        self.llm_var = tk.StringVar(value="Llama 2")
        llm_dropdown = ttk.Combobox(llm_frame, textvariable=self.llm_var, values=["Llama 2"], state="readonly")
        llm_dropdown.pack(side=tk.LEFT)

        # Question Input
        input_frame = tk.Frame(self.master, bg='#2b2b2b')
        input_frame.pack(pady=10, padx=10, fill=tk.X)

        self.question_entry = tk.Entry(input_frame, bg='#3c3f41', fg='white', insertbackground='white')
        self.question_entry.pack(side=tk.LEFT, expand=True, fill=tk.X)

        self.ask_button = tk.Button(input_frame, text="Ask", command=self.ask_question, bg='#3c3f41', fg='white')
        self.ask_button.pack(side=tk.RIGHT, padx=5)

        # Answer Output
        output_frame = tk.Frame(self.master, bg='#2b2b2b')
        output_frame.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

        self.answer_text = tk.Text(output_frame, wrap=tk.WORD, bg='#3c3f41', fg='white', state='disabled')
        self.answer_text.pack(expand=True, fill=tk.BOTH)

    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.file_label.config(text=file_path.split("/")[-1])
            self.pdf_status_label.config(text="Processing PDF...", fg='yellow')
            self.backend.load_pdf(file_path, self.update_progress, self.pdf_processing_complete)

    def update_progress(self, value, message):
        self.progress_bar['value'] = value
        self.progress_label.config(text=message)
        self.master.update_idletasks()

    def pdf_processing_complete(self, success, message):
        if success:
            self.pdf_status_label.config(text=message, fg='green')
            self.setup_rag_pipeline()
        else:
            self.pdf_status_label.config(text=message, fg='red')
        self.master.update_idletasks()

    def setup_rag_pipeline(self):
        try:
            self.pdf_status_label.config(text="Setting up RAG pipeline...", fg='yellow')
            self.master.update_idletasks()
            self.backend.setup_rag_pipeline(self.stream_callback)
            self.pdf_status_label.config(text="RAG pipeline set up successfully", fg='green')
        except Exception as e:
            self.pdf_status_label.config(text=f"Error setting up RAG pipeline: {str(e)}", fg='red')
        self.master.update_idletasks()

    def stream_callback(self, token):
        self.answer_text.config(state='normal')
        self.answer_text.insert(tk.END, token)
        self.answer_text.see(tk.END)
        self.answer_text.config(state='disabled')
        self.master.update_idletasks()

    def ask_question(self):
        question = self.question_entry.get()
        if question:
            self.ask_button.config(state='disabled')
            self.answer_text.config(state='normal')
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(tk.END, "Processing your question...\n\n")
            self.answer_text.config(state='disabled')
            self.master.update_idletasks()
            self.backend.ask_question(question, self.process_answer)
        else:
            self.answer_text.config(state='normal')
            self.answer_text.delete(1.0, tk.END)
            self.answer_text.insert(tk.END, "Please enter a question.")
            self.answer_text.config(state='disabled')

    def process_answer(self, answer):
        self.answer_text.config(state='normal')
        self.answer_text.delete(1.0, tk.END)
        self.answer_text.insert(tk.END, answer)
        self.answer_text.config(state='disabled')
        self.ask_button.config(state='normal')
        self.master.update_idletasks()

if __name__ == "__main__":
    root = tk.Tk()
    app = LLMUI(root)
    root.mainloop()