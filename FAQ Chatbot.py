import tkinter as tk 
from tkinter import scrolledtext, messagebox, ttk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class FAQChatbot:
    def __init__(self, root):
        self.root = root
        root.title("🤖 AI FAQ Chatbot")
        root.geometry("750x600")
        root.configure(bg='#2c3e50')        
        self.faq_data = [
            {"question": "What is artificial intelligence?", 
             "answer": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans."},
            {"question": "How does machine learning work?", 
             "answer": "Machine learning uses algorithms to parse data, learn from it, and make predictions or decisions without being explicitly programmed."},
            {"question": "What is deep learning?", 
             "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to analyze various factors of data."},
            {"question": "What programming languages are used for AI?", 
             "answer": "Python is the most popular language for AI development, followed by R, Java, and C++."},
            {"question": "What is natural language processing?", 
             "answer": "NLP is a branch of AI that helps computers understand, interpret, and manipulate human language."},
            {"question": "How can I get started with AI?", 
             "answer": "Start by learning Python, then study machine learning fundamentals, and practice with projects using libraries like TensorFlow or PyTorch."},
            {"question": "What is computer vision?", 
             "answer": "Computer vision is a field of AI that enables computers to interpret and make decisions based on visual data from the world."},
            {"question": "What are neural networks?", 
             "answer": "Neural networks are computing systems inspired by the human brain that recognize patterns and solve common problems in AI."},
            {"question": "What is the difference between AI and machine learning?", 
             "answer": "AI is the broader concept of machines being able to carry out tasks smartly, while ML is a subset of AI that focuses on machines learning from data."},
            {"question": "How accurate are AI models?", 
             "answer": "AI model accuracy varies widely depending on the task, data quality, and algorithm, ranging from 70% to over 99% for specific applications."}
            ]
        self.preprocess_data()
        self.setup_gui()    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        return text    
    def preprocess_data(self):
        self.questions = [item['question'] for item in self.faq_data]
        self.answers = [item['answer'] for item in self.faq_data]
        self.processed_questions = [self.preprocess_text(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.processed_questions)    
    def find_best_match(self, user_question):
        processed_user_q = self.preprocess_text(user_question)
        user_vector = self.vectorizer.transform([processed_user_q])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        best_match_idx = np.argmax(similarities)
        best_score = similarities[0, best_match_idx]
        if best_score > 0.3:
            return self.answers[best_match_idx], best_score
        else:
            return None, best_score    
    def setup_gui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2c3e50')
        style.configure('TLabel', background='#2c3e50', foreground='#ecf0f1', font=('Arial', 10))
        style.configure('TButton', font=('Arial', 11, 'bold'))
        style.configure('TLabelframe', background='#34495e')
        style.configure('TLabelframe.Label', background='#34495e', foreground='#ecf0f1', font=('Arial', 11, 'bold'))
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        header = tk.Label(main_container, text="🤖 AI FAQ CHATBOT", 
                         font=("Arial", 18, "bold"), bg='#2c3e50', fg='#ffd700')
        header.pack(pady=(0, 20))
        chat_frame = ttk.LabelFrame(main_container, text="💬 CHAT HISTORY")
        chat_frame.pack(fill='both', expand=True, pady=10)        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, 
                                                     state="disabled", font=('Arial', 10, 'bold'),
                                                     bg='#34495e', fg='#ecf0f1',
                                                     insertbackground='#3498db')
        self.chat_display.pack(fill='both', expand=True, padx=10, pady=10)
        input_frame = ttk.Frame(main_container)
        input_frame.pack(fill='x', pady=10)
        input_frame.columnconfigure(0, weight=1)        
        self.user_input = ttk.Entry(input_frame, font=('Arial', 11))
        self.user_input.grid(row=0, column=0, sticky='ew', padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.process_input())        
        send_btn = tk.Button(input_frame, text="🚀 SEND", 
                           command=self.process_input,
                           font=('Arial', 11, 'bold'),
                           bg='#e74c3c', fg='white',
                           activebackground='#c0392b',
                           activeforeground='white',
                           relief='raised',
                           bd=0,
                           padx=20,
                           pady=8,
                           cursor='hand2')
        send_btn.grid(row=0, column=1)
        help_text = "Ask me about AI, machine learning, neural networks, or related topics!"
        help_label = tk.Label(main_container, text=help_text, font=('Arial', 9), 
                            bg='#2c3e50', fg='#bdc3c7')
        help_label.pack(pady=5)        
        self.add_message("Bot", "Hello! I'm an NLP-powered FAQ chatbot. Ask me anything about Artificial Intelligence!")    
    def process_input(self):
        user_text = self.user_input.get().strip()
        if not user_text:
            return
        self.add_message("You", user_text)
        self.user_input.delete(0, tk.END)
        response, score = self.find_best_match(user_text)
        if response:
            self.add_message("Bot", response)
            self.add_message("Bot", f"(Confidence: {score:.2%})")
        else:
            self.add_message("Bot", "I'm not sure about that. Could you try rephrasing your question?")
            self.add_message("Bot", "I specialize in AI and machine learning topics.")    
    def add_message(self, sender, message):
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"{sender}: {message}\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)
        
def main():
    root = tk.Tk()
    chatbot = FAQChatbot(root)
    root.mainloop()
if __name__ == "__main__":
    main()