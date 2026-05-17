

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog, font
import threading
import os
import sys
import json
import time
import psutil
from datetime import datetime
from pathlib import Path

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import (
        DeepNovaAI, Transformer, ProductionTokenizer, ModelArgs,
        load_model, save_model, get_best_device, get_memory_info,
        cleanup_memory, logger, torch, F
    )
    MODEL_OK = True
except ImportError as e:
    print(f"Error: {e}")
    MODEL_OK = False
    sys.exit(1)


class DevTool:
    """Professional development tool for DeepNova AI"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DeepNova DevTool - AI Development Environment")
        self.root.geometry("1400x900")
        self.root.configure(bg="#1e1e1e")
        
        # Model instances
        self.model = None
        self.tokenizer = None
        self.args = None
        self.deepnova = None
        self.model_loaded = False
        self.loading = False
        
        # Testing
        self.testing = False
        self.benchmark_results = []
        
        # Settings
        self.temp = tk.DoubleVar(value=0.7)
        self.top_p = tk.DoubleVar(value=0.9)
        self.top_k = tk.IntVar(value=50)
        self.max_tokens = tk.IntVar(value=500)
        self.repeat_penalty = tk.DoubleVar(value=1.0)
        self.batch_size = tk.IntVar(value=1)
        self.model_size = tk.StringVar(value="lite")
        self.use_cuda = tk.BooleanVar(value=True)
        self.use_amp = tk.BooleanVar(value=True)
        self.debug_mode = tk.BooleanVar(value=False)
        
        # Build UI
        self.setup_styles()
        self.create_menu()
        self.create_layout()
        self.setup_shortcuts()
        
        # Start monitoring
        self.start_monitoring()
        
        # Auto load
        self.root.after(1000, self.auto_load)
    
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Dark theme for dev tool
        self.colors = {
            'bg': '#1e1e1e',
            'fg': '#d4d4d4',
            'accent': '#007acc',
            'accent_hover': '#005a9e',
            'sidebar': '#252526',
            'border': '#3c3c3c',
            'terminal': '#0e0e0e',
            'terminal_fg': '#4ec9b0',
            'error': '#f48771',
            'warning': '#ce9178',
            'success': '#6a9955',
            'info': '#9cdcfe'
        }
        
        style.configure("Editor.TFrame", background=self.colors['bg'])
        style.configure("Sidebar.TFrame", background=self.colors['sidebar'])
        style.configure("Terminal.TFrame", background=self.colors['terminal'])
        
        style.configure("Accent.TButton", background=self.colors['accent'],
                       foreground="white", borderwidth=0)
        style.map("Accent.TButton",
                 background=[('active', self.colors['accent_hover'])])
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root, bg=self.colors['bg'], fg=self.colors['fg'])
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self.load_model_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Model", command=self.save_model, accelerator="Ctrl+S")
        file_menu.add_command(label="Export Model", command=self.export_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Model Info", command=self.show_model_info)
        model_menu.add_command(label="Parameter Count", command=self.show_params)
        model_menu.add_command(label="Layer Visualization", command=self.visualize_layers)
        model_menu.add_separator()
        model_menu.add_command(label="Run Benchmark", command=self.run_benchmark)
        model_menu.add_command(label="Profile Model", command=self.profile_model)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Code Generator", command=self.code_generator)
        tools_menu.add_command(label="Prompt Optimizer", command=self.prompt_optimizer)
        tools_menu.add_command(label="Dataset Viewer", command=self.dataset_viewer)
        tools_menu.add_separator()
        tools_menu.add_command(label="Memory Cleanup", command=self.cleanup)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Toggle Sidebar", command=self.toggle_sidebar)
        view_menu.add_command(label="Toggle Terminal", command=self.toggle_terminal)
        view_menu.add_separator()
        view_menu.add_checkbutton(label="Debug Mode", variable=self.debug_mode)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self.show_docs)
        help_menu.add_command(label="API Reference", command=self.show_api)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_layout(self):
        """Create main layout"""
        # Main paned window
        self.main_pane = tk.PanedWindow(self.root, bg=self.colors['bg'],
                                        sashwidth=4, sashrelief="flat")
        self.main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left: Editor area
        left_frame = tk.Frame(self.main_pane, bg=self.colors['bg'])
        self.main_pane.add(left_frame, width=800)
        self.create_editor_area(left_frame)
        
        # Right: Sidebar
        right_frame = tk.Frame(self.main_pane, bg=self.colors['sidebar'])
        self.main_pane.add(right_frame, width=400)
        self.create_sidebar(right_frame)
        
        # Bottom: Terminal
        self.terminal_frame = tk.Frame(self.root, bg=self.colors['terminal'], height=200)
        self.create_terminal()
    
    def create_editor_area(self, parent):
        """Create code editor area"""
        # Notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Chat tab
        self.chat_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.chat_tab, text="Chat")
        self.create_chat_tab(self.chat_tab)
        
        # Code editor tab
        self.editor_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.editor_tab, text="Code Editor")
        self.create_editor_tab(self.editor_tab)
        
        # Console tab
        self.console_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.console_tab, text="Python Console")
        self.create_console_tab(self.console_tab)
        
        # Testing tab
        self.test_tab = tk.Frame(self.notebook, bg=self.colors['bg'])
        self.notebook.add(self.test_tab, text="Testing")
        self.create_test_tab(self.test_tab)
    
    def create_chat_tab(self, parent):
        """Create chat interface"""
        # Split pane
        chat_pane = tk.PanedWindow(parent, bg=self.colors['bg'], orient=tk.HORIZONTAL)
        chat_pane.pack(fill=tk.BOTH, expand=True)
        
        # Chat history
        history_frame = tk.Frame(chat_pane, bg=self.colors['bg'])
        chat_pane.add(history_frame, width=500)
        
        self.chat_display = scrolledtext.ScrolledText(
            history_frame, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.colors['bg'], fg=self.colors['fg'],
            insertbackground=self.colors['accent']
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure tags
        self.chat_display.tag_config("user", foreground=self.colors['info'])
        self.chat_display.tag_config("assistant", foreground=self.colors['success'])
        self.chat_display.tag_config("system", foreground=self.colors['warning'])
        self.chat_display.tag_config("error", foreground=self.colors['error'])
        
        # Variables panel
        vars_frame = tk.Frame(chat_pane, bg=self.colors['sidebar'])
        chat_pane.add(vars_frame, width=250)
        
        tk.Label(vars_frame, text="Session Variables", bg=self.colors['sidebar'],
                fg=self.colors['fg'], font=("Segoe UI", 10, "bold")).pack(pady=5)
        
        self.vars_text = scrolledtext.ScrolledText(vars_frame, height=15,
                                                    bg=self.colors['sidebar'],
                                                    fg=self.colors['fg'],
                                                    font=("Consolas", 9))
        self.vars_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input area
        input_frame = tk.Frame(parent, bg=self.colors['bg'], height=100)
        input_frame.pack(fill=tk.X, side=tk.BOTTOM)
        input_frame.pack_propagate(False)
        
        self.input_field = scrolledtext.ScrolledText(
            input_frame, height=3, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.colors['terminal'], fg=self.colors['terminal_fg'],
            insertbackground=self.colors['accent']
        )
        self.input_field.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        btn_frame = tk.Frame(input_frame, bg=self.colors['bg'])
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.send_btn = tk.Button(btn_frame, text="Send", command=self.send_message,
                                  bg=self.colors['accent'], fg="white",
                                  font=("Segoe UI", 9, "bold"), padx=15, pady=3)
        self.send_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_generation,
                                  bg="#f48771", fg="white", padx=15, pady=3,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_chat,
                              bg=self.colors['sidebar'], fg=self.colors['fg'],
                              padx=15, pady=3)
        clear_btn.pack(side=tk.LEFT, padx=2)
    
    def create_editor_tab(self, parent):
        """Create code editor"""
        # Toolbar
        toolbar = tk.Frame(parent, bg=self.colors['sidebar'], height=35)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)
        
        run_btn = tk.Button(toolbar, text="Run Code", command=self.run_code,
                           bg=self.colors['success'], fg="white", padx=10)
        run_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        analyze_btn = tk.Button(toolbar, text="Analyze", command=self.analyze_code,
                               bg=self.colors['accent'], fg="white", padx=10)
        analyze_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        format_btn = tk.Button(toolbar, text="Format", command=self.format_code,
                              bg=self.colors['warning'], fg="white", padx=10)
        format_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Editor
        self.code_editor = scrolledtext.ScrolledText(
            parent, wrap=tk.NONE, font=("Consolas", 11),
            bg=self.colors['terminal'], fg=self.colors['terminal_fg'],
            insertbackground=self.colors['accent']
        )
        self.code_editor.pack(fill=tk.BOTH, expand=True)
        
        # Insert template
        template = '''# DeepNova Code Template
import torch
from model import Transformer, ModelArgs

# Create model
args = ModelArgs.deepseek_v3_lite()
model = Transformer(args)

# Print model info
print(f"Model parameters: {model.total_params:,}")

# Test forward pass
dummy_input = torch.randint(0, args.vocab_size, (1, 10))
output, aux_loss, mtp_loss = model(dummy_input)
print(f"Output shape: {output.shape}")
'''
        self.code_editor.insert(tk.END, template)
    
    def create_console_tab(self, parent):
        """Create Python console"""
        self.console = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.colors['terminal'], fg=self.colors['terminal_fg']
        )
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.insert(tk.END, "Python 3.11.0 (DeepNova Console)\n")
        self.console.insert(tk.END, "Type code and press Ctrl+Enter to execute\n")
        self.console.insert(tk.END, ">>> ")
        self.console.mark_set("prompt", "insert-1c")
        self.console.mark_gravity("prompt", tk.LEFT)
        
        self.console.bind("<Control-Return>", self.execute_console)
        self.console.bind("<Key>", self.handle_console_key)
        
        self.console_history = []
        self.history_pos = 0
    
    def create_test_tab(self, parent):
        """Create testing interface"""
        # Test controls
        control_frame = tk.Frame(parent, bg=self.colors['sidebar'], height=80)
        control_frame.pack(fill=tk.X)
        control_frame.pack_propagate(False)
        
        tk.Label(control_frame, text="Test Type:", bg=self.colors['sidebar'],
                fg=self.colors['fg']).pack(side=tk.LEFT, padx=10)
        
        self.test_type = ttk.Combobox(control_frame, values=[
            "Forward Pass", "Generation", "Memory Test", "Speed Benchmark",
            "Accuracy Test", "Load Test"
        ], state="readonly", width=20)
        self.test_type.set("Forward Pass")
        self.test_type.pack(side=tk.LEFT, padx=5)
        
        run_test_btn = tk.Button(control_frame, text="Run Test", command=self.run_test,
                                bg=self.colors['accent'], fg="white", padx=15)
        run_test_btn.pack(side=tk.LEFT, padx=10)
        
        # Test output
        self.test_output = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.colors['terminal'], fg=self.colors['terminal_fg']
        )
        self.test_output.pack(fill=tk.BOTH, expand=True)
    
    def create_sidebar(self, parent):
        """Create sidebar with controls"""
        # Model control
        model_frame = tk.LabelFrame(parent, text="Model Control", bg=self.colors['sidebar'],
                                    fg=self.colors['fg'], font=("Segoe UI", 10, "bold"))
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(model_frame, text="Size:", bg=self.colors['sidebar'],
                fg=self.colors['fg']).pack(pady=2)
        size_combo = ttk.Combobox(model_frame, textvariable=self.model_size,
                                   values=["lite", "base", "enhanced"], state="readonly")
        size_combo.pack(pady=2)
        
        self.load_btn = tk.Button(model_frame, text="Load Model", command=self.load_model,
                                  bg=self.colors['accent'], fg="white")
        self.load_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Model info
        self.model_info = tk.Text(model_frame, height=8, width=30,
                                  bg=self.colors['terminal'], fg=self.colors['terminal_fg'],
                                  font=("Consolas", 8))
        self.model_info.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters
        param_frame = tk.LabelFrame(parent, text="Generation Parameters", bg=self.colors['sidebar'],
                                    fg=self.colors['fg'], font=("Segoe UI", 10, "bold"))
        param_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Temperature
        f = tk.Frame(param_frame, bg=self.colors['sidebar'])
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Temperature:", bg=self.colors['sidebar'], fg=self.colors['fg']).pack(side=tk.LEFT)
        tk.Scale(f, from_=0.1, to=1.5, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.temp, bg=self.colors['sidebar'], length=150).pack(side=tk.RIGHT)
        
        # Top-p
        f = tk.Frame(param_frame, bg=self.colors['sidebar'])
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Top-p:", bg=self.colors['sidebar'], fg=self.colors['fg']).pack(side=tk.LEFT)
        tk.Scale(f, from_=0.5, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self.top_p, bg=self.colors['sidebar'], length=150).pack(side=tk.RIGHT)
        
        # Top-k
        f = tk.Frame(param_frame, bg=self.colors['sidebar'])
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Top-k:", bg=self.colors['sidebar'], fg=self.colors['fg']).pack(side=tk.LEFT)
        tk.Spinbox(f, from_=1, to=100, textvariable=self.top_k, width=8).pack(side=tk.RIGHT)
        
        # Max tokens
        f = tk.Frame(param_frame, bg=self.colors['sidebar'])
        f.pack(fill=tk.X, pady=2)
        tk.Label(f, text="Max Tokens:", bg=self.colors['sidebar'], fg=self.colors['fg']).pack(side=tk.LEFT)
        tk.Spinbox(f, from_=50, to=2000, textvariable=self.max_tokens, width=8).pack(side=tk.RIGHT)
        
        # Hardware
        hw_frame = tk.LabelFrame(parent, text="Hardware", bg=self.colors['sidebar'],
                                 fg=self.colors['fg'], font=("Segoe UI", 10, "bold"))
        hw_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.device_label = tk.Label(hw_frame, text="Device: Detecting...",
                                     bg=self.colors['sidebar'], fg=self.colors['info'])
        self.device_label.pack(pady=2)
        
        self.memory_label = tk.Label(hw_frame, text="Memory: ...",
                                     bg=self.colors['sidebar'], fg=self.colors['info'])
        self.memory_label.pack(pady=2)
    
    def create_terminal(self):
        """Create terminal at bottom"""
        self.terminal_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.terminal_frame.pack_propagate(False)
        
        # Terminal toolbar
        term_toolbar = tk.Frame(self.terminal_frame, bg=self.colors['sidebar'], height=25)
        term_toolbar.pack(fill=tk.X)
        term_toolbar.pack_propagate(False)
        
        tk.Label(term_toolbar, text="Terminal", bg=self.colors['sidebar'],
                fg=self.colors['fg']).pack(side=tk.LEFT, padx=5)
        
        clear_term_btn = tk.Button(term_toolbar, text="Clear", command=self.clear_terminal,
                                   bg=self.colors['accent'], fg="white", height=1)
        clear_term_btn.pack(side=tk.RIGHT, padx=5)
        
        # Terminal text
        self.terminal = scrolledtext.ScrolledText(
            self.terminal_frame, wrap=tk.WORD, font=("Consolas", 9),
            bg=self.colors['terminal'], fg=self.colors['terminal_fg'],
            height=10
        )
        self.terminal.pack(fill=tk.BOTH, expand=True)
        
        self.log("DeepNova DevTool v1.0 initialized")
        self.log(f"Python: {sys.version}")
        self.log(f"PyTorch: {torch.__version__}")
        self.log(f"Device: {get_best_device()}")
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind("<Control-o>", lambda e: self.load_model_dialog())
        self.root.bind("<Control-s>", lambda e: self.save_model())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<Control-l>", lambda e: self.clear_chat())
        self.input_field.bind("<Control-Return>", lambda e: self.send_message())
    
    def log(self, msg, level="INFO"):
        """Log to terminal"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {"INFO": self.colors['info'], "WARN": self.colors['warning'],
                  "ERROR": self.colors['error'], "SUCCESS": self.colors['success']}
        self.terminal.insert(tk.END, f"[{timestamp}] [{level}] {msg}\n", level)
        self.terminal.see(tk.END)
    
    def start_monitoring(self):
        """Start system monitoring"""
        def monitor():
            try:
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1e9
                    self.memory_label.config(text=f"GPU Memory: {mem:.2f} GB")
                device = get_best_device()
                self.device_label.config(text=f"Device: {device.upper()}")
            except:
                pass
            self.root.after(2000, monitor)
        
        self.root.after(2000, monitor)
    
    def auto_load(self):
        """Auto load model"""
        if not self.model_loaded and not self.loading:
            self.load_model()
    
    def load_model_dialog(self):
        """Open file dialog to load model"""
        path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Model files", "*.pt *.safetensors"), ("All", "*.*")]
        )
        if path:
            self.model_path = os.path.dirname(path)
            self.load_model()
    
    def load_model(self):
        """Load model"""
        if self.loading:
            return
        
        self.loading = True
        self.log("Loading model...", "INFO")
        self.load_btn.config(state=tk.DISABLED, text="Loading...")
        
        def load():
            try:
                if self.model_size.get() == "enhanced":
                    args = ModelArgs.enhanced_full()
                elif self.model_size.get() == "base":
                    args = ModelArgs()
                else:
                    args = ModelArgs.deepseek_v3_lite()
                
                args.device = get_best_device() if self.use_cuda.get() else "cpu"
                
                self.tokenizer = ProductionTokenizer()
                self.model = Transformer(args)
                self.model = self.model.to(torch.device(args.device))
                self.model.eval()
                
                self.deepnova = DeepNovaAI(self.model, self.tokenizer, args)
                self.model_loaded = True
                
                self.root.after(0, self.on_load_success)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_load_error(str(e)))
            
            finally:
                self.loading = False
        
        threading.Thread(target=load, daemon=True).start()
    
    def on_load_success(self):
        """Handle load success"""
        self.log("Model loaded successfully!", "SUCCESS")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        
        info = self.deepnova.get_model_info()
        self.model_info.delete(1.0, tk.END)
        self.model_info.insert(tk.END, f"Params: {info.get('total_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Active: {info.get('active_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Layers: {info.get('n_layers', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Experts: {info.get('n_experts', 'N/A')}")
    
    def on_load_error(self, error):
        """Handle load error"""
        self.log(f"Load failed: {error}", "ERROR")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
    
    def send_message(self):
        """Send chat message"""
        if not self.model_loaded:
            self.log("Model not loaded", "WARN")
            return
        
        if self.generating:
            return
        
        msg = self.input_field.get(1.0, tk.END).strip()
        if not msg:
            return
        
        self.input_field.delete(1.0, tk.END)
        self.add_chat_message("user", msg)
        
        self.generating = True
        self.stop_flag = False
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        def generate():
            try:
                resp = self.deepnova.chat(
                    msg,
                    max_new_tokens=self.max_tokens.get(),
                    temperature=self.temp.get()
                )
                if not self.stop_flag:
                    self.root.after(0, lambda: self.add_chat_message("assistant", resp))
            except Exception as e:
                self.root.after(0, lambda: self.add_chat_message("system", f"Error: {e}"))
            finally:
                self.root.after(0, self.on_generation_done)
        
        threading.Thread(target=generate, daemon=True).start()
    
    def on_generation_done(self):
        """Handle generation done"""
        self.generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
    
    def stop_generation(self):
        """Stop generation"""
        self.stop_flag = True
        self.log("Generation stopped", "WARN")
    
    def add_chat_message(self, role, msg):
        """Add message to chat"""
        self.chat_display.config(state=tk.NORMAL)
        time = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"\n[{time}] {role.upper()}:\n", role)
        self.chat_display.insert(tk.END, f"{msg}\n")
        self.chat_display.insert(tk.END, "-" * 60 + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.log("Chat cleared", "INFO")
    
    def run_code(self):
        """Run code from editor"""
        code = self.code_editor.get(1.0, tk.END)
        self.log("Executing code...", "INFO")
        
        def execute():
            try:
                exec(code, {"__name__": "__main__", "torch": torch, "model": self.model})
                self.log("Code executed successfully", "SUCCESS")
            except Exception as e:
                self.log(f"Code error: {e}", "ERROR")
        
        threading.Thread(target=execute, daemon=True).start()
    
    def analyze_code(self):
        """Analyze code"""
        code = self.code_editor.get(1.0, tk.END)
        lines = len(code.split('\n'))
        chars = len(code)
        self.log(f"Code analysis: {lines} lines, {chars} characters", "INFO")
    
    def format_code(self):
        """Format code (basic)"""
        code = self.code_editor.get(1.0, tk.END)
        # Simple formatting
        lines = code.split('\n')
        formatted = []
        indent = 0
        for line in lines:
            stripped = line.strip()
            if stripped.endswith(':'):
                formatted.append('    ' * indent + stripped)
                indent += 1
            elif stripped and stripped[0] in '}])':
                indent = max(0, indent - 1)
                formatted.append('    ' * indent + stripped)
            else:
                formatted.append('    ' * indent + stripped)
        self.code_editor.delete(1.0, tk.END)
        self.code_editor.insert(1.0, '\n'.join(formatted))
        self.log("Code formatted", "INFO")
    
    def execute_console(self, event=None):
        """Execute Python console command"""
        code = self.console.get("prompt", tk.END).strip()
        if not code:
            return
        
        self.console.insert(tk.END, "\n")
        self.console_history.append(code)
        self.history_pos = len(self.console_history)
        
        def execute():
            try:
                result = eval(code, {"__builtins__": __builtins__, "torch": torch})
                if result is not None:
                    self.console.insert(tk.END, f"{result}\n")
            except:
                try:
                    exec(code, {"__builtins__": __builtins__, "torch": torch})
                except Exception as e:
                    self.console.insert(tk.END, f"Error: {e}\n")
            
            self.console.insert(tk.END, ">>> ")
            self.console.mark_set("prompt", "insert-1c")
            self.console.see(tk.END)
        
        threading.Thread(target=execute, daemon=True).start()
    
    def handle_console_key(self, event):
        """Handle console key presses"""
        if event.keysym == "Up":
            if self.history_pos > 0:
                self.history_pos -= 1
                self.console.delete("prompt", tk.END)
                self.console.insert("prompt", self.console_history[self.history_pos])
            return "break"
        elif event.keysym == "Down":
            if self.history_pos < len(self.console_history) - 1:
                self.history_pos += 1
                self.console.delete("prompt", tk.END)
                self.console.insert("prompt", self.console_history[self.history_pos])
            return "break"
    
    def run_test(self):
        """Run selected test"""
        if not self.model_loaded:
            self.log("Model not loaded", "WARN")
            return
        
        test_type = self.test_type.get()
        self.log(f"Running test: {test_type}", "INFO")
        self.test_output.delete(1.0, tk.END)
        
        def run():
            try:
                if test_type == "Forward Pass":
                    self.test_forward_pass()
                elif test_type == "Generation":
                    self.test_generation()
                elif test_type == "Memory Test":
                    self.test_memory()
                elif test_type == "Speed Benchmark":
                    self.test_speed()
                elif test_type == "Load Test":
                    self.test_load()
            except Exception as e:
                self.root.after(0, lambda: self.test_output.insert(tk.END, f"Error: {e}\n"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def test_forward_pass(self):
        """Test forward pass"""
        self.test_output.insert(tk.END, "Testing forward pass...\n")
        dummy = torch.randint(0, self.model.vocab_size, (2, 32))
        start = time.time()
        logits, aux, mtp = self.model(dummy)
        elapsed = time.time() - start
        self.test_output.insert(tk.END, f"Forward pass: {elapsed*1000:.2f}ms\n")
        self.test_output.insert(tk.END, f"Logits shape: {logits.shape}\n")
        self.test_output.insert(tk.END, f"Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB\n")
    
    def test_generation(self):
        """Test generation"""
        self.test_output.insert(tk.END, "Testing generation...\n")
        prompt = "Hello, how are you?"
        start = time.time()
        response = self.deepnova.chat(prompt, max_new_tokens=50)
        elapsed = time.time() - start
        self.test_output.insert(tk.END, f"Prompt: {prompt}\n")
        self.test_output.insert(tk.END, f"Response: {response[:200]}...\n")
        self.test_output.insert(tk.END, f"Time: {elapsed:.2f}s\n")
    
    def test_memory(self):
        """Memory test"""
        self.test_output.insert(tk.END, "Memory test...\n")
        mem_info = get_memory_info()
        for k, v in mem_info.items():
            self.test_output.insert(tk.END, f"{k}: {v}\n")
    
    def test_speed(self):
        """Speed benchmark"""
        self.test_output.insert(tk.END, "Speed benchmark...\n")
        times = []
        for i in range(10):
            dummy = torch.randint(0, self.model.vocab_size, (1, 10))
            start = time.time()
            _ = self.model(dummy)
            times.append(time.time() - start)
        avg = sum(times) / len(times)
        self.test_output.insert(tk.END, f"Average forward time: {avg*1000:.2f}ms\n")
        self.test_output.insert(tk.END, f"Min: {min(times)*1000:.2f}ms, Max: {max(times)*1000:.2f}ms\n")
    
    def test_load(self):
        """Load test"""
        self.test_output.insert(tk.END, "Load test...\n")
        import psutil
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        self.test_output.insert(tk.END, f"CPU Usage: {cpu}%\n")
        self.test_output.insert(tk.END, f"RAM Usage: {mem.percent}%\n")
        if torch.cuda.is_available():
            self.test_output.insert(tk.END, f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB\n")
    
    def run_benchmark(self):
        """Run comprehensive benchmark"""
        self.log("Running benchmark...", "INFO")
        self.test_type.set("Speed Benchmark")
        self.run_test()
    
    def profile_model(self):
        """Profile model"""
        self.log("Profiling model...", "INFO")
        if self.model:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.log(f"Total params: {total:,}", "INFO")
            self.log(f"Trainable: {trainable:,}", "INFO")
    
    def show_model_info(self):
        """Show model information"""
        if self.deepnova:
            info = self.deepnova.get_model_info()
            msg = json.dumps(info, indent=2)
            messagebox.showinfo("Model Info", msg)
    
    def show_params(self):
        """Show parameter count"""
        if self.model:
            total = count_parameters(self.model)
            messagebox.showinfo("Parameters", f"Total parameters: {total:,}")
    
    def visualize_layers(self):
        """Visualize model layers"""
        if self.model:
            layers = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'in_features'):
                    layers.append(f"{name}: {type(module).__name__}")
            text = "\n".join(layers[:50])
            messagebox.showinfo("Layers", text)
    
    def code_generator(self):
        """Generate code for model usage"""
        template = '''# DeepNova Model Usage Example
from model import DeepNovaAI, ProductionTokenizer, ModelArgs

# Load model
args = ModelArgs.deepseek_v3_lite()
tokenizer = ProductionTokenizer()
model = Transformer(args)
deepnova = DeepNovaAI(model, tokenizer, args)

# Chat
response = deepnova.chat("Hello!")
print(response)

# Learn from text
deepnova.learn("Important information to remember")

# Recall knowledge
results = deepnova.recall("What did I learn?")
for r in results:
    print(r['summary'])
'''
        self.code_editor.delete(1.0, tk.END)
        self.code_editor.insert(1.0, template)
        self.log("Code template generated", "INFO")
    
    def prompt_optimizer(self):
        """Optimize prompts"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Prompt Optimizer")
        dialog.geometry("600x400")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text="Enter your prompt:", bg=self.colors['bg'],
                fg=self.colors['fg']).pack(pady=10)
        
        prompt_entry = scrolledtext.ScrolledText(dialog, height=5, font=("Consolas", 10))
        prompt_entry.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(dialog, text="Optimized prompt:", bg=self.colors['bg'],
                fg=self.colors['fg']).pack(pady=5)
        
        result_text = scrolledtext.ScrolledText(dialog, height=8, font=("Consolas", 10))
        result_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        
        def optimize():
            prompt = prompt_entry.get(1.0, tk.END).strip()
            # Basic optimization
            optimized = f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"
            result_text.delete(1.0, tk.END)
            result_text.insert(1.0, optimized)
        
        tk.Button(dialog, text="Optimize", command=optimize,
                 bg=self.colors['accent'], fg="white", padx=20).pack(pady=10)
    
    def dataset_viewer(self):
        """View dataset"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Dataset Viewer")
        dialog.geometry("800x500")
        dialog.configure(bg=self.colors['bg'])
        
        path = filedialog.askdirectory(title="Select dataset directory")
        if path:
            files = list(Path(path).glob("*.txt")) + list(Path(path).glob("*.json"))
            text = "\n".join([f.name for f in files[:100]])
            tk.Text(dialog, wrap=tk.WORD, font=("Consolas", 10)).pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def cleanup(self):
        """Clean up memory"""
        cleanup_memory()
        self.log("Memory cleaned", "SUCCESS")
    
    def save_model(self):
        """Save model"""
        if not self.model_loaded:
            self.log("No model to save", "WARN")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pt")
        if path:
            save_model(self.model, os.path.dirname(path))
            self.log(f"Model saved to {path}", "SUCCESS")
    
    def export_model(self):
        """Export model"""
        self.save_model()
    
    def toggle_sidebar(self):
        """Toggle sidebar visibility"""
        # Not implemented in this version
        pass
    
    def toggle_terminal(self):
        """Toggle terminal visibility"""
        if self.terminal_frame.winfo_viewable():
            self.terminal_frame.pack_forget()
        else:
            self.terminal_frame.pack(fill=tk.X, side=tk.BOTTOM)
    
    def clear_terminal(self):
        """Clear terminal"""
        self.terminal.delete(1.0, tk.END)
    
    def show_docs(self):
        """Show documentation"""
        doc_text = """DeepNova DevTool Documentation

Features:
- Model loading and management
- Interactive chat with AI
- Code editor with Python support
- Python console for testing
- Performance benchmarking
- Memory profiling
- Model testing suite
- Code generation tools

Keyboard Shortcuts:
- Ctrl+O: Load model
- Ctrl+S: Save model
- Ctrl+Q: Quit
- Ctrl+Enter: Send message (chat/console)

For more information, visit the documentation."""
        messagebox.showinfo("Documentation", doc_text)
    
    def show_api(self):
        """Show API reference"""
        api_text = """DeepNova API Reference

ModelArgs:
- dim: Model dimension
- n_layers: Number of layers
- n_heads: Number of attention heads
- vocab_size: Vocabulary size

Transformer:
- forward(input_ids): Forward pass
- generate(input_ids): Text generation

DeepNovaAI:
- chat(message): Chat with AI
- learn(text): Learn from text
- recall(query): Recall knowledge
- get_stats(): Get statistics

For full API, see model.py"""
        messagebox.showinfo("API Reference", api_text)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """DeepNova DevTool v1.0

Professional development environment for DeepNova AI

Features:
- MoE Transformer architecture
- Multi-expert routing
- Intelligent memory system
- Code generation tools

Created for AI developers and researchers

License: MIT"""
        messagebox.showinfo("About", about_text)
    
    def run(self):
        """Run application"""
        def on_close():
            cleanup_memory()
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_close)
        self.root.mainloop()


def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    app = DevTool()
    app.run()