#!/usr/bin/env python3

import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog, ttk
import threading
import os
import sys
from datetime import datetime

# Import DeepNova
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import (
        DeepNovaAI, Transformer, ProductionTokenizer, ModelArgs,
        load_model, get_best_device, cleanup_memory, logger, torch
    )
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure model.py is in the same directory")
    sys.exit(1)


class ChatGUI:
    """Simple chat GUI - white background, blue buttons"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DeepNova Chat")
        self.root.geometry("1000x700")
        self.root.configure(bg="white")
        
        # Model
        self.model = None
        self.tokenizer = None
        self.args = None
        self.deepnova = None
        self.model_loaded = False
        self.loading = False
        
        # Chat state
        self.generating = False
        self.stop_flag = False
        
        # Settings
        self.temp = tk.DoubleVar(value=0.7)
        self.max_tokens = tk.IntVar(value=500)
        
        # Create UI
        self.setup_ui()
        self.setup_shortcuts()
        
        # Auto load
        self.root.after(500, self.auto_load)
    
    def setup_ui(self):
        """Create all UI elements"""
        
        # Main container
        main = tk.Frame(self.root, bg="white")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.create_header(main)
        
        # Chat area
        self.create_chat(main)
        
        # Input area
        self.create_input(main)
        
        # Sidebar
        self.create_sidebar()
        
        # Status bar
        self.create_status(main)
    
    def create_header(self, parent):
        """Create header bar"""
        header = tk.Frame(parent, bg="#2196F3", height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text="DeepNova Chat", 
                font=("Arial", 16, "bold"),
                bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=20)
        
        self.status_indicator = tk.Label(header, text="● Offline",
                                         font=("Arial", 9),
                                         bg="#2196F3", fg="#FF9800")
        self.status_indicator.pack(side=tk.RIGHT, padx=20)
        
        tk.Button(header, text="Settings", command=self.toggle_sidebar,
                 font=("Arial", 9), bg="#1976D2", fg="white",
                 relief="flat", padx=10, pady=5).pack(side=tk.RIGHT, padx=10)
    
    def create_chat(self, parent):
        """Create chat display area"""
        # Chat frame
        chat_frame = tk.Frame(parent, bg="white")
        chat_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Chat text
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, font=("Arial", 10),
            bg="white", fg="#333333", relief="solid", borderwidth=1
        )
        self.chat_text.pack(fill=tk.BOTH, expand=True)
        self.chat_text.config(state=tk.DISABLED)
        
        # Text tags
        self.chat_text.tag_config("user", foreground="#2196F3", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("bot", foreground="#4CAF50", font=("Arial", 10, "bold"))
        self.chat_text.tag_config("system", foreground="#FF9800", font=("Arial", 9, "italic"))
        
        # Welcome
        self.add_message("system", "Welcome! I'm DeepNova. How can I help you?")
    
    def create_input(self, parent):
        """Create input area"""
        input_frame = tk.Frame(parent, bg="white")
        input_frame.pack(fill=tk.X)
        
        # Input text
        self.input_text = scrolledtext.ScrolledText(
            input_frame, height=4, wrap=tk.WORD, font=("Arial", 10),
            bg="white", fg="#333333", relief="solid", borderwidth=1
        )
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Buttons
        btn_frame = tk.Frame(input_frame, bg="white")
        btn_frame.pack(fill=tk.X)
        
        self.send_btn = tk.Button(btn_frame, text="Send", command=self.send,
                                  font=("Arial", 10, "bold"),
                                  bg="#2196F3", fg="white", padx=20, pady=5,
                                  relief="flat", cursor="hand2")
        self.send_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop,
                                  font=("Arial", 10),
                                  bg="#FF9800", fg="white", padx=20, pady=5,
                                  relief="flat", cursor="hand2", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        tk.Label(btn_frame, text="Ctrl+Enter to send", font=("Arial", 8),
                bg="white", fg="#999").pack(side=tk.RIGHT)
    
    def create_sidebar(self):
        """Create settings sidebar"""
        self.sidebar = tk.Toplevel(self.root)
        self.sidebar.title("Settings")
        self.sidebar.geometry("300x400")
        self.sidebar.configure(bg="#F5F5F5")
        self.sidebar.withdraw()
        
        # Model section
        group = tk.LabelFrame(self.sidebar, text="Model", bg="#F5F5F5", fg="#333", font=("Arial", 10, "bold"))
        group.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(group, text="Model Size:", bg="#F5F5F5").pack(pady=5)
        self.model_size = ttk.Combobox(group, values=["lite", "base"], state="readonly")
        self.model_size.set("lite")
        self.model_size.pack(pady=5)
        
        self.load_btn = tk.Button(group, text="Load Model", command=self.load_model,
                                  bg="#2196F3", fg="white", relief="flat", pady=5)
        self.load_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Generation
        group2 = tk.LabelFrame(self.sidebar, text="Generation", bg="#F5F5F5", fg="#333", font=("Arial", 10, "bold"))
        group2.pack(fill=tk.X, padx=10, pady=10)
        
        # Temperature
        frame = tk.Frame(group2, bg="#F5F5F5")
        frame.pack(fill=tk.X, pady=5)
        tk.Label(frame, text="Temperature:", bg="#F5F5F5").pack(side=tk.LEFT)
        scale = tk.Scale(frame, from_=0.1, to=1.5, resolution=0.05,
                        orient=tk.HORIZONTAL, variable=self.temp, bg="#F5F5F5", length=150)
        scale.pack(side=tk.RIGHT)
        
        # Max tokens
        frame2 = tk.Frame(group2, bg="#F5F5F5")
        frame2.pack(fill=tk.X, pady=5)
        tk.Label(frame2, text="Max Tokens:", bg="#F5F5F5").pack(side=tk.LEFT)
        tk.Scale(frame2, from_=100, to=2000, resolution=50,
                orient=tk.HORIZONTAL, variable=self.max_tokens, bg="#F5F5F5", length=150).pack(side=tk.RIGHT)
        
        # Close
        tk.Button(self.sidebar, text="Close", command=self.toggle_sidebar,
                 bg="#999", fg="white", relief="flat", pady=5).pack(pady=10)
    
    def create_status(self, parent):
        """Create status bar"""
        status = tk.Frame(parent, bg="#F5F5F5", height=25)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        status.pack_propagate(False)
        
        self.status_label = tk.Label(status, text="Ready", bg="#F5F5F5", fg="#666", font=("Arial", 8))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.mem_label = tk.Label(status, text="", bg="#F5F5F5", fg="#666", font=("Arial", 8))
        self.mem_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.input_text.bind("<Control-Return>", lambda e: self.send())
        self.input_text.bind("<Command-Return>", lambda e: self.send())
    
    def add_message(self, sender, msg):
        """Add message to chat"""
        self.chat_text.config(state=tk.NORMAL)
        
        time = datetime.now().strftime("%H:%M")
        
        if sender == "user":
            self.chat_text.insert(tk.END, f"\n[{time}] You:\n", "user")
        elif sender == "bot":
            self.chat_text.insert(tk.END, f"\n[{time}] DeepNova:\n", "bot")
        else:
            self.chat_text.insert(tk.END, f"\n[{time}] System:\n", "system")
        
        self.chat_text.insert(tk.END, f"{msg}\n")
        self.chat_text.insert(tk.END, "-"*50 + "\n")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)
    
    def update_status(self, msg, is_error=False):
        """Update status bar"""
        self.status_label.config(text=msg)
        if is_error:
            self.status_label.config(fg="red")
            self.root.after(3000, lambda: self.status_label.config(fg="#666"))
    
    def toggle_sidebar(self):
        """Show/hide sidebar"""
        if self.sidebar.winfo_viewable():
            self.sidebar.withdraw()
        else:
            self.sidebar.deiconify()
    
    def auto_load(self):
        """Auto load model"""
        if not self.model_loaded and not self.loading:
            self.load_model()
    
    def load_model(self):
        """Load model"""
        if self.loading:
            return
        
        self.loading = True
        self.update_status("Loading model...")
        self.status_indicator.config(text="● Loading", fg="#FF9800")
        self.load_btn.config(state=tk.DISABLED, text="Loading...")
        
        def load():
            try:
                # Create config
                if self.model_size.get() == "lite":
                    self.args = ModelArgs.deepseek_v3_lite()
                else:
                    self.args = ModelArgs()
                
                self.args.device = get_best_device()
                
                # Create tokenizer and model
                self.tokenizer = ProductionTokenizer()
                self.model = Transformer(self.args)
                self.model = self.model.to(torch.device(self.args.device))
                self.model.eval()
                
                # Create assistant
                self.deepnova = DeepNovaAI(self.model, self.tokenizer, self.args)
                
                self.model_loaded = True
                self.root.after(0, self.on_load_success)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_load_error(str(e)))
            
            finally:
                self.loading = False
        
        threading.Thread(target=load, daemon=True).start()
    
    def on_load_success(self):
        """Handle load success"""
        self.update_status("Model ready")
        self.status_indicator.config(text="● Ready", fg="#4CAF50")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        self.add_message("system", "Model loaded successfully!")
    
    def on_load_error(self, error):
        """Handle load error"""
        self.model_loaded = False
        self.update_status(f"Error: {error[:30]}", True)
        self.status_indicator.config(text="● Error", fg="#F44336")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        messagebox.showerror("Error", f"Failed to load model:\n{error}")
    
    def send(self):
        """Send message"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if self.generating:
            return
        
        msg = self.input_text.get(1.0, tk.END).strip()
        if not msg:
            return
        
        # Clear input
        self.input_text.delete(1.0, tk.END)
        
        # Add user message
        self.add_message("user", msg)
        
        # Handle commands
        if msg.startswith('/'):
            self.handle_cmd(msg)
            return
        
        # Generate response
        self.generating = True
        self.stop_flag = False
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("Generating...")
        
        def generate():
            try:
                response = self.deepnova.chat(
                    msg,
                    max_new_tokens=self.max_tokens.get(),
                    temperature=self.temp.get()
                )
                
                if not self.stop_flag:
                    self.root.after(0, lambda: self.add_message("bot", response))
                
            except Exception as e:
                self.root.after(0, lambda: self.add_message("system", f"Error: {e}"))
            
            finally:
                self.root.after(0, self.on_done)
        
        threading.Thread(target=generate, daemon=True).start()
    
    def on_done(self):
        """Handle generation done"""
        self.generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.update_status("Ready")
    
    def stop(self):
        """Stop generation"""
        self.stop_flag = True
        self.update_status("Stopping...")
        self.add_message("system", "Generation stopped")
    
    def handle_cmd(self, cmd):
        """Handle commands"""
        parts = cmd[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if command == "clear":
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)
            self.add_message("system", "Chat cleared")
        
        elif command == "stats":
            if self.deepnova:
                stats = self.deepnova.get_stats()
                msg = f"Messages: {stats.get('total_messages', 0)}\nTokens: {stats.get('total_tokens_generated', 0)}"
                self.add_message("system", msg)
        
        elif command == "learn":
            if arg:
                result = self.deepnova.learn(arg)
                if result.get('success'):
                    self.add_message("system", f"Learned: {result['summary'][:100]}")
                else:
                    self.add_message("system", f"Failed: {result.get('error', 'Unknown')}")
            else:
                self.add_message("system", "Usage: /learn <text>")
        
        elif command == "recall":
            if arg:
                results = self.deepnova.recall(arg, top_k=3)
                if results:
                    self.add_message("system", f"Found {len(results)} items:")
                    for r in results:
                        self.add_message("system", f"- {r['summary']}")
                else:
                    self.add_message("system", "No results")
            else:
                self.add_message("system", "Usage: /recall <query>")
        
        elif command == "help":
            help_text = """Commands:
/clear - Clear chat
/stats - Show stats
/learn <text> - Learn new info
/recall <query> - Search knowledge
/help - Show this help"""
            self.add_message("system", help_text)
        
        else:
            self.add_message("system", f"Unknown: {command}. Type /help")
    
    def run(self):
        """Run the app"""
        def on_close():
            cleanup_memory()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


if __name__ == "__main__":
    app = ChatGUI()
    app.run()