

import argparse
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
from datetime import datetime

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import (
        DeepNovaAI, Transformer, ProductionTokenizer, ModelArgs,
        load_model, get_best_device, get_memory_info,
        cleanup_memory, logger, torch, interactive_mode,
        learn_mode, recall_mode, stats_mode, clear_mode,
        export_mode, list_mode, generate_mode, train_mode
    )
    MODEL_OK = True
except ImportError as e:
    print(f"Error: {e}")
    MODEL_OK = False
    sys.exit(1)


class DeepNovaApp:
    """Desktop application - white background, blue buttons"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DeepNova AI")
        self.root.geometry("1100x750")
        self.root.configure(bg="#ffffff")
        self.root.minsize(900, 600)
        
        # Variables
        self.model = None
        self.tokenizer = None
        self.args = None
        self.deepnova = None
        self.model_loaded = False
        self.loading = False
        self.generating = False
        self.stop_flag = False
        
        # Settings
        self.temp = tk.DoubleVar(value=0.7)
        self.max_tokens = tk.IntVar(value=500)
        self.model_size = tk.StringVar(value="lite")
        
        # Create UI
        self.setup_ui()
        self.setup_shortcuts()
        
        # Auto load
        self.root.after(500, self.auto_load)
        
        # Update memory periodically
        self.update_memory()
    
    def setup_ui(self):
        """Create all UI elements"""
        
        # Main container
        main = tk.Frame(self.root, bg="#ffffff")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top bar
        self.create_top_bar(main)
        
        # Main content area
        content = tk.Frame(main, bg="#ffffff")
        content.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Chat area
        left = tk.Frame(content, bg="#ffffff")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.create_chat_area(left)
        self.create_input_area(left)
        
        # Right: Sidebar
        right = tk.Frame(content, bg="#f5f5f5", width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right.pack_propagate(False)
        
        self.create_sidebar(right)
        
        # Status bar
        self.create_status_bar(main)
    
    def create_top_bar(self, parent):
        """Create top bar"""
        top = tk.Frame(parent, bg="#2196f3", height=50)
        top.pack(fill=tk.X)
        top.pack_propagate(False)
        
        title = tk.Label(top, text="DeepNova AI", 
                        font=("Segoe UI", 16, "bold"),
                        bg="#2196f3", fg="white")
        title.pack(side=tk.LEFT, padx=20, pady=10)
        
        version = tk.Label(top, text="v5.0", font=("Segoe UI", 10),
                          bg="#2196f3", fg="white")
        version.pack(side=tk.LEFT, padx=(0, 20))
        
        self.status_dot = tk.Label(top, text="●", font=("Segoe UI", 12),
                                   bg="#2196f3", fg="#ff9800")
        self.status_dot.pack(side=tk.RIGHT, padx=(0, 5))
        
        self.status_text = tk.Label(top, text="Not Ready", font=("Segoe UI", 9),
                                    bg="#2196f3", fg="white")
        self.status_text.pack(side=tk.RIGHT, padx=(0, 20))
    
    def create_chat_area(self, parent):
        """Create chat display area"""
        # Chat display
        self.chat = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Segoe UI", 10),
            bg="#ffffff", fg="#333333", relief="flat",
            borderwidth=0, highlightthickness=0
        )
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.config(state=tk.DISABLED)
        
        # Text tags
        self.chat.tag_config("user", foreground="#2196f3", font=("Segoe UI", 10, "bold"))
        self.chat.tag_config("assistant", foreground="#4caf50", font=("Segoe UI", 10, "bold"))
        self.chat.tag_config("system", foreground="#ff9800", font=("Segoe UI", 9, "italic"))
        
        # Welcome
        self.add_message("system", "Welcome! I am DeepNova. How can I help you?")
    
    def create_input_area(self, parent):
        """Create input area"""
        frame = tk.Frame(parent, bg="#ffffff")
        frame.pack(fill=tk.X, pady=(10, 0))
        
        # Input text
        self.input = scrolledtext.ScrolledText(
            frame, height=4, wrap=tk.WORD, font=("Segoe UI", 10),
            bg="#ffffff", fg="#333333", relief="solid",
            borderwidth=1, highlightthickness=0
        )
        self.input.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        # Button frame
        btn_frame = tk.Frame(frame, bg="#ffffff")
        btn_frame.pack(fill=tk.X)
        
        self.send_btn = tk.Button(btn_frame, text="Send", command=self.send,
                                  font=("Segoe UI", 10, "bold"),
                                  bg="#2196f3", fg="white", padx=25, pady=5,
                                  relief="flat", cursor="hand2")
        self.send_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop,
                                  font=("Segoe UI", 10),
                                  bg="#ff9800", fg="white", padx=25, pady=5,
                                  relief="flat", cursor="hand2", state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        
        hint = tk.Label(btn_frame, text="Ctrl+Enter to send", font=("Segoe UI", 8),
                        bg="#ffffff", fg="#999999")
        hint.pack(side=tk.RIGHT)
    
    def create_sidebar(self, parent):
        """Create sidebar"""
        # Model section
        model_frame = tk.LabelFrame(parent, text="Model", font=("Segoe UI", 10, "bold"),
                                    bg="#f5f5f5", fg="#333333", relief="flat")
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model size
        size_frame = tk.Frame(model_frame, bg="#f5f5f5")
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Size:", bg="#f5f5f5", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        size_combo = ttk.Combobox(size_frame, textvariable=self.model_size,
                                   values=["lite", "base"], state="readonly", width=10)
        size_combo.pack(side=tk.RIGHT)
        
        # Load button
        self.load_btn = tk.Button(model_frame, text="Load Model", command=self.load_model,
                                  bg="#2196f3", fg="white", font=("Segoe UI", 9),
                                  padx=10, pady=5, relief="flat")
        self.load_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Model info
        self.model_info = tk.Text(model_frame, height=4, width=25,
                                  bg="white", fg="#333333", font=("Segoe UI", 8),
                                  relief="flat", padx=5, pady=5)
        self.model_info.pack(padx=10, pady=5, fill=tk.X)
        
        # Generation section
        gen_frame = tk.LabelFrame(parent, text="Generation", font=("Segoe UI", 10, "bold"),
                                  bg="#f5f5f5", fg="#333333", relief="flat")
        gen_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Temperature
        temp_frame = tk.Frame(gen_frame, bg="#f5f5f5")
        temp_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(temp_frame, text="Temperature:", bg="#f5f5f5", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        scale = tk.Scale(temp_frame, from_=0.1, to=1.5, resolution=0.05,
                        orient=tk.HORIZONTAL, variable=self.temp,
                        bg="#f5f5f5", length=120)
        scale.pack(side=tk.RIGHT)
        
        # Max tokens
        tokens_frame = tk.Frame(gen_frame, bg="#f5f5f5")
        tokens_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(tokens_frame, text="Max Tokens:", bg="#f5f5f5", font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(tokens_frame, from_=100, to=2000, resolution=50,
                orient=tk.HORIZONTAL, variable=self.max_tokens,
                bg="#f5f5f5", length=120).pack(side=tk.RIGHT)
        
        # Knowledge section
        know_frame = tk.LabelFrame(parent, text="Knowledge", font=("Segoe UI", 10, "bold"),
                                   bg="#f5f5f5", fg="#333333", relief="flat")
        know_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn1 = tk.Button(know_frame, text="Learn from File", command=self.learn_file,
                         bg="#4caf50", fg="white", font=("Segoe UI", 9),
                         padx=10, pady=5, relief="flat")
        btn1.pack(fill=tk.X, padx=10, pady=5)
        
        btn2 = tk.Button(know_frame, text="Recall Knowledge", command=self.recall,
                         bg="#2196f3", fg="white", font=("Segoe UI", 9),
                         padx=10, pady=5, relief="flat")
        btn2.pack(fill=tk.X, padx=10, pady=5)
        
        btn3 = tk.Button(know_frame, text="Statistics", command=self.stats,
                         bg="#ff9800", fg="white", font=("Segoe UI", 9),
                         padx=10, pady=5, relief="flat")
        btn3.pack(fill=tk.X, padx=10, pady=5)
        
        btn4 = tk.Button(know_frame, text="Clear Memory", command=self.clear_memory,
                         bg="#f44336", fg="white", font=("Segoe UI", 9),
                         padx=10, pady=5, relief="flat")
        btn4.pack(fill=tk.X, padx=10, pady=5)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status = tk.Frame(parent, bg="#f5f5f5", height=25)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        status.pack_propagate(False)
        
        self.status_msg = tk.Label(status, text="Ready", bg="#f5f5f5", fg="#666666",
                                   font=("Segoe UI", 8))
        self.status_msg.pack(side=tk.LEFT, padx=10)
        
        self.mem_label = tk.Label(status, text="", bg="#f5f5f5", fg="#666666",
                                  font=("Segoe UI", 8))
        self.mem_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.input.bind("<Control-Return>", lambda e: self.send())
        self.input.bind("<Command-Return>", lambda e: self.send())
    
    def add_message(self, sender, msg):
        """Add message to chat"""
        self.chat.config(state=tk.NORMAL)
        
        time = datetime.now().strftime("%H:%M:%S")
        
        if sender == "user":
            self.chat.insert(tk.END, f"\n[{time}] You:\n", "user")
        elif sender == "assistant":
            self.chat.insert(tk.END, f"\n[{time}] DeepNova:\n", "assistant")
        else:
            self.chat.insert(tk.END, f"\n[{time}] System:\n", "system")
        
        self.chat.insert(tk.END, f"{msg}\n")
        self.chat.insert(tk.END, "-" * 50 + "\n")
        self.chat.see(tk.END)
        self.chat.config(state=tk.DISABLED)
    
    def update_status(self, msg, is_error=False):
        """Update status bar"""
        self.status_msg.config(text=msg)
        if is_error:
            self.status_msg.config(fg="red")
            self.root.after(3000, lambda: self.status_msg.config(fg="#666666"))
    
    def update_memory(self):
        """Update memory info"""
        try:
            if self.deepnova:
                stats = self.deepnova.get_stats()
                self.mem_label.config(
                    text=f"Msgs: {stats.get('total_messages', 0)} | "
                         f"Learned: {stats.get('learning', {}).get('total_learned', 0)}"
                )
        except:
            pass
        self.root.after(5000, self.update_memory)
    
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
        self.status_dot.config(fg="#ff9800")
        self.status_text.config(text="Loading")
        self.load_btn.config(state=tk.DISABLED, text="Loading...")
        
        def load():
            try:
                if self.model_size.get() == "lite":
                    args = ModelArgs.deepseek_v3_lite()
                else:
                    args = ModelArgs()
                
                args.device = get_best_device()
                
                tokenizer = ProductionTokenizer()
                model = Transformer(args)
                model = model.to(torch.device(args.device))
                model.eval()
                
                self.deepnova = DeepNovaAI(model, tokenizer, args)
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
        self.status_dot.config(fg="#4caf50")
        self.status_text.config(text="Ready")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        
        info = self.deepnova.get_model_info()
        self.model_info.delete(1.0, tk.END)
        self.model_info.insert(tk.END, f"Params: {info.get('total_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Active: {info.get('active_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Experts: {info.get('n_experts', 'N/A')}")
        
        self.add_message("system", "Model loaded successfully!")
    
    def on_load_error(self, error):
        """Handle load error"""
        self.model_loaded = False
        self.update_status(f"Error: {error[:30]}", True)
        self.status_dot.config(fg="#f44336")
        self.status_text.config(text="Error")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        messagebox.showerror("Error", f"Failed to load model:\n{error}")
    
    def send(self):
        """Send message"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load model first")
            return
        
        if self.generating:
            return
        
        msg = self.input.get(1.0, tk.END).strip()
        if not msg:
            return
        
        self.input.delete(1.0, tk.END)
        self.add_message("user", msg)
        
        if msg.startswith('/'):
            self.handle_command(msg)
            return
        
        self.generating = True
        self.stop_flag = False
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_status("Generating...")
        
        def generate():
            try:
                resp = self.deepnova.chat(
                    msg,
                    max_new_tokens=self.max_tokens.get(),
                    temperature=self.temp.get()
                )
                if not self.stop_flag:
                    self.root.after(0, lambda: self.add_message("assistant", resp))
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
        self.update_memory()
    
    def stop(self):
        """Stop generation"""
        self.stop_flag = True
        self.update_status("Stopping...")
        self.add_message("system", "Generation stopped")
    
    def handle_command(self, cmd):
        """Handle slash commands"""
        parts = cmd[1:].split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if command == "clear":
            self.chat.config(state=tk.NORMAL)
            self.chat.delete(1.0, tk.END)
            self.chat.config(state=tk.DISABLED)
            self.add_message("system", "Chat cleared")
        
        elif command == "stats":
            self.stats()
        
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
            help_text = "Commands:\n/clear - Clear chat\n/stats - Show stats\n/learn <text> - Learn\n/recall <query> - Search\n/help - This help"
            self.add_message("system", help_text)
        
        else:
            self.add_message("system", f"Unknown: {command}. Type /help")
    
    def learn_file(self):
        """Learn from file"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load model first")
            return
        
        path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[("Text files", "*.txt *.md"), ("All", "*.*")]
        )
        if path:
            self.update_status(f"Learning from {os.path.basename(path)}...")
            
            def learn():
                try:
                    results = self.deepnova.learn_from_file(path)
                    success = len([r for r in results if r.get('success')])
                    self.root.after(0, lambda: self.add_message("system", f"Learned {success} segments from {os.path.basename(path)}"))
                except Exception as e:
                    self.root.after(0, lambda: self.add_message("system", f"Error: {e}"))
                finally:
                    self.root.after(0, lambda: self.update_status("Ready"))
            
            threading.Thread(target=learn, daemon=True).start()
    
    def recall(self):
        """Recall knowledge"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load model first")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Recall Knowledge")
        dialog.geometry("500x400")
        dialog.configure(bg="#ffffff")
        
        tk.Label(dialog, text="Enter query:", bg="#ffffff", font=("Segoe UI", 11)).pack(pady=10)
        
        entry = tk.Text(dialog, height=3, font=("Segoe UI", 10), wrap=tk.WORD)
        entry.pack(fill=tk.X, padx=20, pady=5)
        
        result = scrolledtext.ScrolledText(dialog, height=12, font=("Segoe UI", 10),
                                           wrap=tk.WORD, bg="#f5f5f5")
        result.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        def search():
            query = entry.get(1.0, tk.END).strip()
            if not query:
                return
            result.delete(1.0, tk.END)
            result.insert(tk.END, "Searching...\n")
            
            def search_thread():
                try:
                    res = self.deepnova.recall(query, top_k=5)
                    self.root.after(0, lambda: display(res))
                except Exception as e:
                    self.root.after(0, lambda: result.insert(tk.END, f"Error: {e}"))
            
            def display(res):
                result.delete(1.0, tk.END)
                if res:
                    for i, r in enumerate(res):
                        result.insert(tk.END, f"{i+1}. Score: {r['score']:.2f}\n")
                        result.insert(tk.END, f"   {r['summary']}\n")
                        if r.get('source'):
                            result.insert(tk.END, f"   Source: {r['source']}\n")
                        result.insert(tk.END, "\n")
                else:
                    result.insert(tk.END, "No results found.")
            
            threading.Thread(target=search_thread, daemon=True).start()
        
        tk.Button(dialog, text="Search", command=search,
                 bg="#2196f3", fg="white", font=("Segoe UI", 10),
                 padx=20, pady=5, relief="flat").pack(pady=10)
    
    def stats(self):
        """Show statistics"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load model first")
            return
        
        stats = self.deepnova.get_stats()
        
        text = f"""
DeepNova Statistics
{'='*40}

Model: {stats.get('name', 'N/A')}
Version: {stats.get('version', 'N/A')}
Uptime: {stats.get('uptime_seconds', 0):.0f} sec

Messages: {stats.get('total_messages', 0)}
Tokens Generated: {stats.get('total_tokens_generated', 0)}
Tokens Saved: {stats.get('total_tokens_saved', 0)}

Memory:
  Short-term: {stats.get('memory', {}).get('short_term_messages', 0)}
  Important Facts: {stats.get('memory', {}).get('important_facts', 0)}
  Entities: {stats.get('memory', {}).get('entities_tracked', 0)}

Knowledge:
  Learned: {stats.get('learning', {}).get('total_learned', 0)}
  Graph Nodes: {stats.get('learning', {}).get('knowledge_graph_nodes', 0)}

Features: {', '.join(stats.get('active_features', []))}
"""
        messagebox.showinfo("Statistics", text)
    
    def clear_memory(self):
        """Clear memory"""
        if self.deepnova:
            self.deepnova.clear_context(keep_important=True)
            self.add_message("system", "Memory cleared (keeping important facts)")
            self.update_memory()
    
    def run(self):
        """Run application"""
        def on_close():
            cleanup_memory()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


def resolve_model_args(model_size: str = "lite", device: str = None) -> ModelArgs:
    if model_size == 'max' or model_size == '671b':
        model_args = ModelArgs.deepnova_max()
    elif model_size == 'pro':
        model_args = ModelArgs.deepnova_pro()
    elif model_size in ('instans', 'parallel'):
        model_args = ModelArgs.deepnova_instans()
    elif model_size == 'lite':
        model_args = ModelArgs.deepnova_lite()
    else:
        model_args = ModelArgs()

    if device:
        model_args.device = device
    return model_args


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepNova AI launcher with desktop GUI and CLI subcommands"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    gui_parser = subparsers.add_parser("gui", help="Launch the desktop GUI")
    gui_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    gui_parser.add_argument("--memory-file", default="deepnova_memory.json", help="Memory file path")
    gui_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    gui_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    chat_parser = subparsers.add_parser("chat", help="Interactive CLI chat mode")
    chat_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    chat_parser.add_argument("--memory-file", default="deepnova_memory.json", help="Memory file path")
    chat_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    chat_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")
    chat_parser.add_argument("--assistant-style", choices=["deepnova", "gemini", "claude"], default="deepnova", help="Assistant style hint")

    generate_parser = subparsers.add_parser("generate", help="Generate text from a prompt")
    generate_parser.add_argument("--prompt", required=True, help="Prompt text to generate from")
    generate_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    generate_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    generate_parser.add_argument("--memory-file", default="deepnova_memory.json", help="Memory file path")
    generate_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")
    generate_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    generate_parser.add_argument("--max-tokens", type=int, default=500, help="Maximum new tokens")
    generate_parser.add_argument("--assistant-style", choices=["deepnova", "gemini", "claude"], default="deepnova", help="Assistant style hint")

    tokenize_parser = subparsers.add_parser("tokenize", help="Encode text into token IDs")
    tokenize_parser.add_argument("--text", required=True, help="Text to tokenize")
    tokenize_parser.add_argument("--add-special-tokens", action="store_true", help="Add BOS/EOS tokens")

    detokenize_parser = subparsers.add_parser("detokenize", help="Decode token IDs into text")
    detokenize_parser.add_argument("--ids", required=True, nargs="+", type=int, help="Token IDs to decode")

    learn_parser = subparsers.add_parser("learn", help="Learn from text, file, or directory")
    learn_parser.add_argument("--text", type=str, help="Text to learn")
    learn_parser.add_argument("--file", type=str, help="File path to learn from")
    learn_parser.add_argument("--directory", type=str, help="Directory path to learn from")
    learn_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    learn_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    learn_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    recall_parser = subparsers.add_parser("recall", help="Recall learned knowledge")
    recall_parser.add_argument("--query", required=True, help="Query text to recall")
    recall_parser.add_argument("--top-k", type=int, default=5, help="Number of relevant items")
    recall_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    recall_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    recall_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    stats_parser = subparsers.add_parser("stats", help="Show model statistics")
    stats_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    stats_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    stats_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    clear_parser = subparsers.add_parser("clear", help="Clear conversation/context memory")
    clear_parser.add_argument("--all", action="store_true", help="Clear all memory including facts")
    clear_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    clear_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    clear_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    export_parser = subparsers.add_parser("export", help="Export learned knowledge to file")
    export_parser.add_argument("--output", default="knowledge_export.json", help="Output file path")
    export_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    export_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    export_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    list_parser = subparsers.add_parser("list", help="List learned texts")
    list_parser.add_argument("--limit", type=int, default=20, help="Maximum items to list")
    list_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    list_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    list_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--data", required=True, help="Path to training data")
    train_parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    train_parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint directory")
    train_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    info_parser = subparsers.add_parser("info", help="Show model and device information")
    info_parser.add_argument("--model-path", default=None, help="Path to model checkpoint directory")
    info_parser.add_argument("--device", default=None, help="Device: auto, cpu, cuda, mps")
    info_parser.add_argument("--model-size", default="lite", choices=["lite", "instans", "pro", "max", "base", "large", "671b"], help="Model configuration")

    parser.set_defaults(command="gui")
    return parser


def build_model(args: argparse.Namespace):
    model_args = resolve_model_args(getattr(args, "model_size", "lite"), getattr(args, "device", None))
    model = Transformer(model_args)

    model_path = getattr(args, "model_path", None)
    if model_path and os.path.exists(model_path):
        try:
            model, model_args = load_model(model_path, model_args.device)
            logger.info("Model loaded from %s", model_path)
        except Exception as e:
            logger.warning("Failed to load model from %s, using initialized model: %s", model_path, e)
    return model, model_args


def run_cli_chat(args: argparse.Namespace) -> None:
    model, model_args = build_model(args)
    tokenizer = ProductionTokenizer()
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file=args.memory_file)

    if getattr(args, "assistant_style", "deepnova") != "deepnova":
        deepnova.system_prompt += f"\n\nAssistant style: {args.assistant_style.title()}. Respond in a professional and concise way."

    logger.info("Starting CLI chat mode with style=%s", args.assistant_style)
    interactive_mode(deepnova)


def run_cli_generate(args: argparse.Namespace) -> None:
    model, model_args = build_model(args)
    tokenizer = ProductionTokenizer()
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file=args.memory_file)

    if getattr(args, "assistant_style", "deepnova") != "deepnova":
        deepnova.system_prompt += f"\n\nAssistant style: {args.assistant_style.title()}. Respond in a professional and concise way."

    response = deepnova.chat(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    logger.info("Generated response: %d tokens", len(tokenizer.encode(response, add_special_tokens=False)))
    print(response)


def run_cli_tokenize(args: argparse.Namespace) -> None:
    tokenizer = ProductionTokenizer()
    ids = tokenizer.encode(args.text, add_special_tokens=args.add_special_tokens)
    print("Token IDs:", ids)
    print("Token count:", len(ids))


def run_cli_detokenize(args: argparse.Namespace) -> None:
    tokenizer = ProductionTokenizer()
    text = tokenizer.decode(args.ids)
    print(text)


def run_cli_info(args: argparse.Namespace) -> None:
    model_args = resolve_model_args(getattr(args, "model_size", "lite"), getattr(args, "device", None))
    print("Model information:")
    print("  Model size:", getattr(args, "model_size", "lite"))
    print("  Device:", model_args.device)
    print("  Vocab size:", getattr(model_args, "vocab_size", "unknown"))
    print("  Max sequence length:", getattr(model_args, "max_seq_len", "unknown"))
    print("  Parallel MoE+Dense:", getattr(model_args, "use_parallel_moe_dense", False))
    print("  GLM enabled:", getattr(model_args, "use_glm", False))
    print("  Adaptive router:", getattr(model_args, "use_adaptive_router", False))
    print("  Multi-token prediction:", getattr(model_args, "use_multi_token_prediction", False))
    model_path = getattr(args, "model_path", None)
    if model_path:
        print("  Model path:", model_path)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    command = getattr(args, "command", "gui") or "gui"

    logger.info("DeepNova App launching command=%s", command)

    if command == "gui":
        app = DeepNovaApp()
        app.run()
    elif command == "chat":
        run_cli_chat(args)
    elif command == "generate":
        run_cli_generate(args)
    elif command == "tokenize":
        run_cli_tokenize(args)
    elif command == "detokenize":
        run_cli_detokenize(args)
    elif command == "learn":
        learn_mode(args)
    elif command == "recall":
        recall_mode(args)
    elif command == "stats":
        stats_mode(args)
    elif command == "clear":
        clear_mode(args)
    elif command == "export":
        export_mode(args)
    elif command == "list":
        list_mode(args)
    elif command == "train":
        train_mode(args)
    elif command == "info":
        run_cli_info(args)
    else:
        parser.print_help()
