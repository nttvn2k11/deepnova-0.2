#!/usr/bin/env python3
"""
DeepNova GUI - Professional White Interface with Blue Buttons
VSCode-style IDE interface for DeepNova AI Assistant
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Import DeepNova from model.py
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import (
        DeepNovaAI, Transformer, ProductionTokenizer, ModelArgs,
        load_model, validate_model_args, get_best_device, get_memory_info,
        cleanup_memory, logger, torch
    )
    DEEPNOVA_AVAILABLE = True
    print("DeepNova modules loaded successfully")
except ImportError as e:
    print(f"Error: Could not import DeepNova modules: {e}")
    print("Make sure model.py is in the same directory")
    DEEPNOVA_AVAILABLE = False
    sys.exit(1)


class DeepNovaGUI:
    """Professional VSCode-style GUI with white background and blue buttons"""
    
    # Color scheme - VSCode inspired
    COLOR_BG = "#FFFFFF"                    # White background
    COLOR_SIDEBAR = "#F3F3F3"               # Light gray sidebar
    COLOR_ACTIVITY_BAR = "#2C2C2C"          # Dark activity bar
    COLOR_BTN = "#007ACC"                    # VSCode blue
    COLOR_BTN_HOVER = "#005A9E"              # Darker blue on hover
    COLOR_BTN_TEXT = "#FFFFFF"               # White text on buttons
    COLOR_TEXT = "#333333"                   # Dark text
    COLOR_TEXT_SECONDARY = "#6A6A6A"         # Secondary text
    COLOR_BORDER = "#E5E5E5"                 # Border color
    COLOR_SUCCESS = "#28A745"                # Green for success
    COLOR_ERROR = "#DC3545"                  # Red for error
    COLOR_WARNING = "#FFC107"                # Yellow for warning
    COLOR_USER_MSG = "#007ACC"               # User message color
    COLOR_AI_MSG = "#28A745"                 # AI message color
    COLOR_SYSTEM_MSG = "#FF8C00"             # System message color
    COLOR_SECONDARY = "#F5F5F5"              # Light gray for secondary elements
    
    def __init__(self, root):
        self.root = root
        self.root.title("DeepNova AI - Professional MoE Assistant")
        self.root.geometry("1400x900")
        self.root.configure(bg=self.COLOR_BG)
        self.root.minsize(1200, 700)
        
        # Model and assistant
        self.model = None
        self.tokenizer = None
        self.args = None
        self.deepnova = None
        self.model_loaded = False
        self.model_loading = False
        
        # Configuration variables
        self.temperature = tk.DoubleVar(value=0.7)
        self.max_tokens = tk.IntVar(value=500)
        self.top_p = tk.DoubleVar(value=0.9)
        self.top_k = tk.IntVar(value=50)
        self.model_size = tk.StringVar(value="lite")
        self.use_enhanced = tk.BooleanVar(value=True)
        self.use_parallel = tk.BooleanVar(value=True)
        self.system_prompt = tk.StringVar(value="You are DeepNova, a professional AI assistant with MoE architecture.")
        
        # Model path
        self.model_path = tk.StringVar(value="")
        
        # State variables
        self.is_generating = False
        self.stop_generation = False
        self.conversation_history = []
        
        # Create interface
        self.setup_styles()
        self.create_menu()
        self.create_main_layout()
        self.setup_bindings()
        
        # Auto load default model
        self.root.after(500, self.auto_load_model)
        
        # Update memory info periodically
        self.update_memory_info()
        
        print("GUI initialized successfully")
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom styles
        style.configure("Blue.TButton", 
                       background=self.COLOR_BTN,
                       foreground=self.COLOR_BTN_TEXT,
                       borderwidth=0,
                       focuscolor="none",
                       font=("Segoe UI", 10))
        style.map("Blue.TButton",
                 background=[('active', self.COLOR_BTN_HOVER)])
        
        style.configure("Sidebar.TFrame", background=self.COLOR_SIDEBAR)
        style.configure("Main.TFrame", background=self.COLOR_BG)
        style.configure("Card.TLabelframe", background=self.COLOR_BG,
                       foreground=self.COLOR_TEXT, relief="solid", borderwidth=1)
        style.configure("Card.TLabelframe.Label", background=self.COLOR_BG,
                       foreground=self.COLOR_TEXT, font=("Segoe UI", 10, "bold"))
    
    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root, bg=self.COLOR_BG, fg=self.COLOR_TEXT)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.COLOR_BG, fg=self.COLOR_TEXT)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self.load_model_dialog, accelerator="Ctrl+O")
        file_menu.add_command(label="Export Conversation", command=self.export_conversation)
        file_menu.add_command(label="Export Knowledge", command=self.export_knowledge)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0, bg=self.COLOR_BG, fg=self.COLOR_TEXT)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Chat", command=self.clear_chat)
        edit_menu.add_command(label="Clear Memory", command=self.clear_memory)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=self.COLOR_BG, fg=self.COLOR_TEXT)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Learn from File", command=self.learn_from_file)
        tools_menu.add_command(label="Learn from Directory", command=self.learn_from_directory)
        tools_menu.add_command(label="Recall Knowledge", command=self.recall_knowledge)
        tools_menu.add_separator()
        tools_menu.add_command(label="Show Statistics", command=self.show_stats)
        tools_menu.add_command(label="Cleanup Memory", command=self.cleanup_memory)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.COLOR_BG, fg=self.COLOR_TEXT)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Shortcuts", command=self.show_shortcuts)
    
    def create_main_layout(self):
        """Create main application layout - VSCode style"""
        
        # Main paned window
        main_pane = tk.PanedWindow(self.root, bg=self.COLOR_BG, sashwidth=6, sashrelief="flat")
        main_pane.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Chat area (main)
        left_frame = tk.Frame(main_pane, bg=self.COLOR_BG)
        main_pane.add(left_frame, width=900)
        self.create_chat_panel(left_frame)
        
        # Right panel - Sidebar
        right_frame = tk.Frame(main_pane, bg=self.COLOR_SIDEBAR)
        main_pane.add(right_frame, width=400)
        self.create_sidebar(right_frame)
        
        # Status bar
        self.create_status_bar()
    
    def create_chat_panel(self, parent):
        """Create chat panel with message display and input"""
        
        # Chat display area
        chat_container = tk.Frame(parent, bg=self.COLOR_BG)
        chat_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header with model info
        header_frame = tk.Frame(chat_container, bg=self.COLOR_BG, height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="DeepNova AI Assistant", 
                font=("Segoe UI", 18, "bold"),
                bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(side=tk.LEFT)
        
        self.model_badge = tk.Label(header_frame, text="● Model: Not Loaded",
                                   font=("Segoe UI", 9), bg=self.COLOR_BG,
                                   fg=self.COLOR_WARNING)
        self.model_badge.pack(side=tk.RIGHT, padx=10)
        
        # Separator
        separator = tk.Frame(chat_container, bg=self.COLOR_BORDER, height=1)
        separator.pack(fill=tk.X, pady=5)
        
        # Chat display (scrolled text)
        self.chat_display = scrolledtext.ScrolledText(
            chat_container, wrap=tk.WORD, font=("Segoe UI", 11),
            bg=self.COLOR_BG, fg=self.COLOR_TEXT,
            insertbackground=self.COLOR_BTN, relief="flat",
            borderwidth=0, highlightthickness=0
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=10)
        self.chat_display.config(state=tk.DISABLED)
        
        # Configure text tags
        self.chat_display.tag_config("user", foreground=self.COLOR_USER_MSG,
                                     font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("assistant", foreground=self.COLOR_AI_MSG,
                                     font=("Segoe UI", 11, "bold"))
        self.chat_display.tag_config("system", foreground=self.COLOR_SYSTEM_MSG,
                                     font=("Segoe UI", 10, "italic"))
        self.chat_display.tag_config("error", foreground=self.COLOR_ERROR,
                                     font=("Segoe UI", 10, "bold"))
        
        # Welcome message
        self.add_welcome_message()
        
        # Input area
        input_frame = tk.Frame(chat_container, bg=self.COLOR_BG)
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input text area
        self.input_field = scrolledtext.ScrolledText(
            input_frame, height=4, wrap=tk.WORD, font=("Segoe UI", 11),
            bg=self.COLOR_BG, fg=self.COLOR_TEXT,
            insertbackground=self.COLOR_BTN, relief="solid",
            borderwidth=1, highlightthickness=1,
            highlightcolor=self.COLOR_BTN, highlightbackground=self.COLOR_BORDER
        )
        self.input_field.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Button frame
        btn_frame = tk.Frame(input_frame, bg=self.COLOR_BG)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Send button
        self.send_btn = tk.Button(btn_frame, text="Send", command=self.send_message,
                                  font=("Segoe UI", 11, "bold"),
                                  bg=self.COLOR_BTN, fg=self.COLOR_BTN_TEXT,
                                  padx=25, pady=12, relief="flat", cursor="hand2",
                                  activebackground=self.COLOR_BTN_HOVER,
                                  activeforeground=self.COLOR_BTN_TEXT)
        self.send_btn.pack(pady=(0, 5))
        
        # Stop button
        self.stop_btn = tk.Button(btn_frame, text="Stop", command=self.stop_response,
                                  font=("Segoe UI", 11),
                                  bg=self.COLOR_WARNING, fg=self.COLOR_BTN_TEXT,
                                  padx=25, pady=12, relief="flat", cursor="hand2",
                                  state=tk.DISABLED)
        self.stop_btn.pack()
        
        # Clear button
        clear_btn = tk.Button(btn_frame, text="Clear", command=self.clear_chat,
                              font=("Segoe UI", 11),
                              bg=self.COLOR_SECONDARY, fg=self.COLOR_TEXT,
                              padx=25, pady=12, relief="flat", cursor="hand2")
        clear_btn.pack(pady=(5, 0))
    
    def create_sidebar(self, parent):
        """Create sidebar with controls - VSCode style"""
        
        # Model section
        model_frame = tk.LabelFrame(parent, text="Model Control", 
                                    font=("Segoe UI", 10, "bold"),
                                    bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                                    relief="solid", borderwidth=1)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model status
        self.status_label = tk.Label(model_frame, text="Status: Not loaded",
                                     font=("Segoe UI", 9), bg=self.COLOR_SIDEBAR,
                                     fg=self.COLOR_WARNING)
        self.status_label.pack(pady=5, padx=10, anchor=tk.W)
        
        # Load button
        self.load_btn = tk.Button(model_frame, text="Load Model", command=self.load_model_dialog,
                                  font=("Segoe UI", 10, "bold"), bg=self.COLOR_BTN,
                                  fg=self.COLOR_BTN_TEXT, padx=10, pady=5,
                                  relief="flat", cursor="hand2")
        self.load_btn.pack(pady=5, padx=10, fill=tk.X)
        
        # Model info
        self.model_info = tk.Text(model_frame, height=6, width=35,
                                  bg=self.COLOR_SECONDARY, fg=self.COLOR_TEXT,
                                  font=("Segoe UI", 8), relief="flat",
                                  borderwidth=0, padx=5, pady=5)
        self.model_info.pack(padx=10, pady=5, fill=tk.X)
        
        # Model config frame
        config_frame = tk.LabelFrame(parent, text="Model Configuration",
                                     font=("Segoe UI", 10, "bold"),
                                     bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                                     relief="solid", borderwidth=1)
        config_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model size
        size_frame = tk.Frame(config_frame, bg=self.COLOR_SIDEBAR)
        size_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(size_frame, text="Model Size:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        size_combo = ttk.Combobox(size_frame, textvariable=self.model_size,
                                   values=["lite", "base", "parallel", "enhanced"],
                                   state="readonly", width=12)
        size_combo.pack(side=tk.RIGHT)
        
        # Enhanced features
        enhanced_frame = tk.Frame(config_frame, bg=self.COLOR_SIDEBAR)
        enhanced_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(enhanced_frame, text="Enhanced Features:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Checkbutton(enhanced_frame, variable=self.use_enhanced,
                      bg=self.COLOR_SIDEBAR, activebackground=self.COLOR_SIDEBAR).pack(side=tk.RIGHT)
        
        # Parallel MoE
        parallel_frame = tk.Frame(config_frame, bg=self.COLOR_SIDEBAR)
        parallel_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(parallel_frame, text="Parallel MoE+Dense:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Checkbutton(parallel_frame, variable=self.use_parallel,
                      bg=self.COLOR_SIDEBAR, activebackground=self.COLOR_SIDEBAR).pack(side=tk.RIGHT)
        
        # Generation parameters
        gen_frame = tk.LabelFrame(parent, text="Generation Parameters",
                                  font=("Segoe UI", 10, "bold"),
                                  bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                                  relief="solid", borderwidth=1)
        gen_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Temperature
        temp_frame = tk.Frame(gen_frame, bg=self.COLOR_SIDEBAR)
        temp_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(temp_frame, text="Temperature:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        temp_scale = tk.Scale(temp_frame, from_=0.1, to=1.5, resolution=0.05,
                              orient=tk.HORIZONTAL, variable=self.temperature,
                              bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                              highlightthickness=0, length=150)
        temp_scale.pack(side=tk.RIGHT)
        
        # Max tokens
        tokens_frame = tk.Frame(gen_frame, bg=self.COLOR_SIDEBAR)
        tokens_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(tokens_frame, text="Max Tokens:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(tokens_frame, from_=50, to=2000, resolution=50,
                orient=tk.HORIZONTAL, variable=self.max_tokens,
                bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                highlightthickness=0, length=150).pack(side=tk.RIGHT)
        
        # Top-p
        topp_frame = tk.Frame(gen_frame, bg=self.COLOR_SIDEBAR)
        topp_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(topp_frame, text="Top-p:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Scale(topp_frame, from_=0.5, to=1.0, resolution=0.05,
                orient=tk.HORIZONTAL, variable=self.top_p,
                bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                highlightthickness=0, length=150).pack(side=tk.RIGHT)
        
        # Top-k
        topk_frame = tk.Frame(gen_frame, bg=self.COLOR_SIDEBAR)
        topk_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(topk_frame, text="Top-k:", bg=self.COLOR_SIDEBAR,
                fg=self.COLOR_TEXT, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        tk.Spinbox(topk_frame, from_=1, to=100, textvariable=self.top_k,
                   width=8, font=("Segoe UI", 9)).pack(side=tk.RIGHT)
        
        # System prompt
        prompt_frame = tk.LabelFrame(parent, text="System Prompt",
                                     font=("Segoe UI", 10, "bold"),
                                     bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                                     relief="solid", borderwidth=1)
        prompt_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, height=6,
                                                      wrap=tk.WORD, font=("Segoe UI", 9),
                                                      bg=self.COLOR_SECONDARY,
                                                      fg=self.COLOR_TEXT, relief="flat")
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.prompt_text.insert(tk.END, self.system_prompt.get())
        
        # Update button
        update_btn = tk.Button(prompt_frame, text="Update Prompt", command=self.update_system_prompt,
                               font=("Segoe UI", 9), bg=self.COLOR_BTN, fg=self.COLOR_BTN_TEXT,
                               padx=10, pady=3, relief="flat", cursor="hand2")
        update_btn.pack(pady=5)
        
        # Knowledge buttons
        knowledge_frame = tk.LabelFrame(parent, text="Knowledge Management",
                                        font=("Segoe UI", 10, "bold"),
                                        bg=self.COLOR_SIDEBAR, fg=self.COLOR_TEXT,
                                        relief="solid", borderwidth=1)
        knowledge_frame.pack(fill=tk.X, padx=10, pady=5)
        
        learn_btn = tk.Button(knowledge_frame, text="Learn from File", command=self.learn_from_file,
                              font=("Segoe UI", 9), bg=self.COLOR_SUCCESS, fg=self.COLOR_BTN_TEXT,
                              padx=10, pady=5, relief="flat", cursor="hand2")
        learn_btn.pack(fill=tk.X, padx=10, pady=5)
        
        recall_btn = tk.Button(knowledge_frame, text="Recall Knowledge", command=self.recall_knowledge,
                               font=("Segoe UI", 9), bg=self.COLOR_BTN, fg=self.COLOR_BTN_TEXT,
                               padx=10, pady=5, relief="flat", cursor="hand2")
        recall_btn.pack(fill=tk.X, padx=10, pady=5)
        
        stats_btn = tk.Button(knowledge_frame, text="Show Statistics", command=self.show_stats,
                              font=("Segoe UI", 9), bg=self.COLOR_WARNING, fg=self.COLOR_BTN_TEXT,
                              padx=10, pady=5, relief="flat", cursor="hand2")
        stats_btn.pack(fill=tk.X, padx=10, pady=5)
    
    def create_status_bar(self):
        """Create status bar at bottom"""
        status_bar = tk.Frame(self.root, bg=self.COLOR_SECONDARY, height=30)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        status_bar.pack_propagate(False)
        
        self.status_message = tk.Label(status_bar, text="Ready",
                                       bg=self.COLOR_SECONDARY, fg=self.COLOR_TEXT,
                                       font=("Segoe UI", 9))
        self.status_message.pack(side=tk.LEFT, padx=10)
        
        self.device_label = tk.Label(status_bar, text="",
                                     bg=self.COLOR_SECONDARY, fg=self.COLOR_TEXT,
                                     font=("Segoe UI", 9))
        self.device_label.pack(side=tk.LEFT, padx=20)
        
        self.memory_label = tk.Label(status_bar, text="",
                                     bg=self.COLOR_SECONDARY, fg=self.COLOR_TEXT,
                                     font=("Segoe UI", 9))
        self.memory_label.pack(side=tk.RIGHT, padx=10)
        
        # Update device info
        self.update_device_info()
    
    def setup_bindings(self):
        """Setup keyboard bindings"""
        self.input_field.bind("<Control-Return>", lambda e: self.send_message())
        self.input_field.bind("<Command-Return>", lambda e: self.send_message())
        self.root.bind("<Control-o>", lambda e: self.load_model_dialog())
        self.root.bind("<Control-l>", lambda e: self.clear_chat())
    
    def add_welcome_message(self):
        """Add welcome message to chat"""
        welcome = """Welcome to DeepNova AI Assistant!

**Features:**
- MoE (Mixture of Experts) architecture with top-k routing
- Parallel MoE + Dense MLP paths
- Adaptive Router with learnable temperature
- Dynamic Depth (layer skipping based on confidence)
- Multi-Token Prediction (MTP)
- Intelligent context memory with compression
- Persistent knowledge storage

**How to use:**
1. Load a model using the sidebar controls
2. Adjust generation parameters (temperature, top-p, etc.)
3. Type your message and press Send or Ctrl+Enter
4. Use the knowledge tools to learn from files

**Available commands:**
- `/clear` - Clear conversation
- `/stats` - Show statistics
- `/learn <text>` - Learn new information
- `/recall <query>` - Recall knowledge

Type your message below to start chatting!"""
        
        self.add_message("system", welcome)
    
    def add_message(self, sender: str, message: str):
        """Add message to chat display"""
        self.chat_display.config(state=tk.NORMAL)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        if sender == "user":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] You:\n", "user")
        elif sender == "assistant":
            self.chat_display.insert(tk.END, f"\n[{timestamp}] DeepNova:\n", "assistant")
        else:
            self.chat_display.insert(tk.END, f"\n[{timestamp}] System:\n", "system")
        
        self.chat_display.insert(tk.END, f"{message}\n")
        self.chat_display.insert(tk.END, "-" * 60 + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def update_device_info(self):
        """Update device info in status bar"""
        try:
            device = get_best_device()
            self.device_label.config(text=f"Device: {device.upper()}")
        except:
            self.device_label.config(text="Device: CPU")
    
    def update_memory_info(self):
        """Update memory info in status bar"""
        try:
            info = get_memory_info()
            if 'gpu_allocated_gb' in info:
                self.memory_label.config(text=f"GPU: {info['gpu_allocated_gb']:.1f}/{info['gpu_total_gb']:.1f}GB | RAM: {info['ram_used_gb']:.0f}/{info['ram_total_gb']:.0f}GB")
            elif 'ram_used_gb' in info:
                self.memory_label.config(text=f"RAM: {info['ram_used_gb']:.0f}/{info['ram_total_gb']:.0f}GB")
        except:
            pass
        self.root.after(3000, self.update_memory_info)
    
    def auto_load_model(self):
        """Auto load default model"""
        if not self.model_loaded and not self.model_loading:
            self.load_model()
    
    def load_model_dialog(self):
        """Show file dialog to select model"""
        file_path = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("Model files", "*.pt *.safetensors"), ("All files", "*.*")]
        )
        if file_path:
            self.model_path.set(os.path.dirname(file_path))
            self.load_model()
    
    def load_model(self):
        """Load model into memory"""
        if self.model_loading:
            return
        
        self.model_loading = True
        self.status_message.config(text="Loading model...")
        self.status_label.config(text="Status: Loading...", fg=self.COLOR_WARNING)
        self.load_btn.config(state=tk.DISABLED, text="Loading...")
        
        def load_thread():
            try:
                # Create config
                if self.model_size.get() == "enhanced":
                    self.args = ModelArgs.enhanced_full()
                elif self.model_size.get() == "parallel" or self.use_parallel.get():
                    self.args = ModelArgs.parallel_moe_dense()
                elif self.model_size.get() == "lite":
                    self.args = ModelArgs.deepseek_v3_lite()
                else:
                    self.args = ModelArgs()
                
                # Apply settings
                if self.use_enhanced.get():
                    self.args.use_parallel_moe_dense = True
                    self.args.use_glm = True
                    self.args.use_adaptive_router = True
                    self.args.use_multi_token_prediction = True
                
                self.args.device = get_best_device()
                self.args.print_model_stats = True
                
                # Create tokenizer and model
                self.tokenizer = ProductionTokenizer()
                self.model = Transformer(self.args)
                
                # Load checkpoint if available
                if self.model_path.get() and os.path.exists(self.model_path.get()):
                    try:
                        self.model, self.args = load_model(self.model_path.get(), self.args.device)
                        self.add_message("system", f"Loaded model from {self.model_path.get()}")
                    except Exception as e:
                        self.add_message("system", f"Warning: Could not load checkpoint - {e}")
                
                self.model = self.model.to(torch.device(self.args.device))
                self.model.eval()
                
                # Create DeepNova AI
                self.deepnova = DeepNovaAI(self.model, self.tokenizer, self.args)
                
                # Update system prompt
                self.update_system_prompt()
                
                self.model_loaded = True
                
                # Update UI
                self.root.after(0, self.on_model_loaded)
                
            except Exception as e:
                self.root.after(0, lambda: self.on_model_load_error(str(e)))
            
            finally:
                self.model_loading = False
        
        thread = threading.Thread(target=load_thread, daemon=True)
        thread.start()
    
    def on_model_loaded(self):
        """Called when model is successfully loaded"""
        self.model_loaded = True
        self.status_message.config(text="Model loaded successfully")
        self.status_label.config(text="Status: Ready", fg=self.COLOR_SUCCESS)
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        self.model_badge.config(text="● Model: Ready", fg=self.COLOR_SUCCESS)
        
        # Update model info
        info = self.deepnova.get_model_info() if hasattr(self.deepnova, 'get_model_info') else {}
        self.model_info.delete(1.0, tk.END)
        self.model_info.insert(tk.END, f"Name: {info.get('model_name', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Params: {info.get('total_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Active: {info.get('active_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Sparsity: {info.get('sparsity', 0):.1%}\n")
        self.model_info.insert(tk.END, f"Layers: {info.get('n_layers', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Experts: {info.get('n_experts', 'N/A')} (top-{info.get('n_activated_experts', 'N/A')})\n")
        
        self.add_message("system", f"Model loaded successfully!\n{info.get('total_params_formatted', 'N/A')} parameters, {info.get('n_experts', 'N/A')} experts")
    
    def on_model_load_error(self, error):
        """Called when model loading fails"""
        self.model_loaded = False
        self.status_message.config(text=f"Error: {error[:50]}")
        self.status_label.config(text="Status: Error", fg=self.COLOR_ERROR)
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        self.model_badge.config(text="● Model: Error", fg=self.COLOR_ERROR)
        self.add_message("system", f"Failed to load model: {error}")
        messagebox.showerror("Error", f"Failed to load model:\n{error}")
    
    def send_message(self):
        """Send user message and get response"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if self.is_generating:
            return
        
        user_input = self.input_field.get(1.0, tk.END).strip()
        if not user_input:
            return
        
        # Clear input
        self.input_field.delete(1.0, tk.END)
        
        # Add user message to chat
        self.add_message("user", user_input)
        
        # Check for commands
        if user_input.startswith('/'):
            self.handle_command(user_input)
            return
        
        # Start generation
        self.is_generating = True
        self.stop_generation = False
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_message.config(text="Generating response...")
        
        def generate_thread():
            try:
                response = self.deepnova.chat(
                    user_input,
                    max_new_tokens=self.max_tokens.get(),
                    temperature=self.temperature.get()
                )
                
                if not self.stop_generation:
                    self.root.after(0, lambda: self.add_message("assistant", response))
                
            except Exception as e:
                self.root.after(0, lambda: self.add_message("system", f"Error: {e}"))
                logger.error(f"Generation error: {e}")
            
            finally:
                self.root.after(0, self.on_generation_complete)
        
        thread = threading.Thread(target=generate_thread, daemon=True)
        thread.start()
    
    def on_generation_complete(self):
        """Called when generation completes"""
        self.is_generating = False
        self.send_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_message.config(text="Ready")
    
    def stop_response(self):
        """Stop ongoing response generation"""
        self.stop_generation = True
        self.status_message.config(text="Stopping...")
        self.add_message("system", "Response generation stopped by user")
    
    def handle_command(self, command):
        """Handle slash commands"""
        cmd_parts = command[1:].split(maxsplit=1)
        cmd = cmd_parts[0].lower()
        arg = cmd_parts[1] if len(cmd_parts) > 1 else ""
        
        if cmd == "clear":
            self.clear_chat()
        elif cmd == "stats":
            self.show_stats()
        elif cmd == "learn":
            if arg:
                result = self.deepnova.learn(arg)
                if result.get('success'):
                    self.add_message("system", f"Learned: {result['summary']}")
                else:
                    self.add_message("system", f"Failed to learn: {result.get('error', 'Unknown error')}")
            else:
                self.add_message("system", "Usage: /learn <text>")
        elif cmd == "recall":
            if arg:
                results = self.deepnova.recall(arg, top_k=3)
                if results:
                    self.add_message("system", f"Found {len(results)} relevant items:")
                    for r in results:
                        self.add_message("system", f"- {r['summary']}")
                else:
                    self.add_message("system", "No relevant knowledge found")
            else:
                self.add_message("system", "Usage: /recall <query>")
        else:
            self.add_message("system", f"Unknown command: {cmd}. Available: /clear, /stats, /learn, /recall")
    
    def update_system_prompt(self):
        """Update system prompt"""
        new_prompt = self.prompt_text.get(1.0, tk.END).strip()
        self.system_prompt.set(new_prompt)
        if self.deepnova:
            self.deepnova.system_prompt = new_prompt
            self.add_message("system", "System prompt updated")
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.add_welcome_message()
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.deepnova:
            self.deepnova.clear_context(keep_important=True)
            self.add_message("system", "Conversation memory cleared (keeping important facts)")
    
    def learn_from_file(self):
        """Learn from selected file"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select File to Learn",
            filetypes=[("Text files", "*.txt *.md"), ("All files", "*.*")]
        )
        if file_path:
            self.status_message.config(text=f"Learning from {os.path.basename(file_path)}...")
            
            def learn_thread():
                try:
                    results = self.deepnova.learn_from_file(file_path)
                    success = len([r for r in results if r.get('success')])
                    self.root.after(0, lambda: self.add_message("system", f"Learned {success} segments from {os.path.basename(file_path)}"))
                except Exception as e:
                    self.root.after(0, lambda: self.add_message("system", f"Error learning from file: {e}"))
                finally:
                    self.root.after(0, lambda: self.status_message.config(text="Ready"))
            
            threading.Thread(target=learn_thread, daemon=True).start()
    
    def learn_from_directory(self):
        """Learn from directory"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        dir_path = filedialog.askdirectory(title="Select Directory to Learn From")
        if dir_path:
            self.status_message.config(text=f"Learning from directory...")
            
            def learn_thread():
                try:
                    results = self.deepnova.learn_from_directory(dir_path)
                    success = len([r for r in results if r.get('success')])
                    self.root.after(0, lambda: self.add_message("system", f"Learned {success} segments from directory"))
                except Exception as e:
                    self.root.after(0, lambda: self.add_message("system", f"Error learning from directory: {e}"))
                finally:
                    self.root.after(0, lambda: self.status_message.config(text="Ready"))
            
            threading.Thread(target=learn_thread, daemon=True).start()
    
    def recall_knowledge(self):
        """Recall knowledge based on query"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        # Create popup dialog for query
        dialog = tk.Toplevel(self.root)
        dialog.title("Recall Knowledge")
        dialog.geometry("500x400")
        dialog.configure(bg=self.COLOR_BG)
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Enter your query:", font=("Segoe UI", 11),
                bg=self.COLOR_BG, fg=self.COLOR_TEXT).pack(pady=10)
        
        query_entry = tk.Text(dialog, height=3, font=("Segoe UI", 10),
                              wrap=tk.WORD, relief="solid", borderwidth=1)
        query_entry.pack(fill=tk.X, padx=20, pady=5)
        
        result_text = scrolledtext.ScrolledText(dialog, height=12, font=("Segoe UI", 10),
                                                 wrap=tk.WORD, bg=self.COLOR_SECONDARY,
                                                 fg=self.COLOR_TEXT, relief="flat")
        result_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        def do_recall():
            query = query_entry.get(1.0, tk.END).strip()
            if not query:
                return
            
            result_text.delete(1.0, tk.END)
            result_text.insert(tk.END, "Searching...\n")
            
            def recall_thread():
                try:
                    results = self.deepnova.recall(query, top_k=5)
                    self.root.after(0, lambda: display_results(results))
                except Exception as e:
                    self.root.after(0, lambda: result_text.insert(tk.END, f"Error: {e}"))
            
            def display_results(results):
                result_text.delete(1.0, tk.END)
                if results:
                    for i, r in enumerate(results):
                        result_text.insert(tk.END, f"{i+1}. Score: {r['score']:.2f}\n")
                        result_text.insert(tk.END, f"   {r['summary']}\n")
                        if r.get('source'):
                            result_text.insert(tk.END, f"   Source: {r['source']}\n")
                        result_text.insert(tk.END, "\n")
                else:
                    result_text.insert(tk.END, "No relevant knowledge found.")
            
            threading.Thread(target=recall_thread, daemon=True).start()
        
        tk.Button(dialog, text="Search", command=do_recall,
                 bg=self.COLOR_BTN, fg=self.COLOR_BTN_TEXT,
                 font=("Segoe UI", 10), padx=20, pady=5,
                 relief="flat", cursor="hand2").pack(pady=10)
    
    def show_stats(self):
        """Show statistics dialog"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        stats = self.deepnova.get_stats()
        
        stats_text = f"""
DeepNova Statistics
{'='*50}

Model Information:
  Name: {stats.get('name', 'N/A')}
  Version: {stats.get('version', 'N/A')}
  Conversation ID: {stats.get('conversation_id', 'N/A')}
  Uptime: {stats.get('uptime_seconds', 0):.0f} seconds

Usage Statistics:
  Total Messages: {stats.get('total_messages', 0)}
  Tokens Generated: {stats.get('total_tokens_generated', 0)}
  Tokens Saved: {stats.get('total_tokens_saved', 0)}
  Chat History: {stats.get('chat_history_length', 0)}

Memory Statistics:
  Short-term Messages: {stats.get('memory', {}).get('short_term_messages', 0)}
  Important Facts: {stats.get('memory', {}).get('important_facts', 0)}
  Entities Tracked: {stats.get('memory', {}).get('entities_tracked', 0)}
  Total Compressions: {stats.get('memory', {}).get('total_compressions', 0)}

Learning Statistics:
  Total Learned: {stats.get('learning', {}).get('total_learned', 0)}
  Knowledge Graph Nodes: {stats.get('learning', {}).get('knowledge_graph_nodes', 0)}
"""
        
        messagebox.showinfo("Statistics", stats_text)
    
    def export_conversation(self):
        """Export conversation to file"""
        file_path = filedialog.asksaveasfilename(
            title="Export Conversation",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            content = self.chat_display.get(1.0, tk.END)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.add_message("system", f"Conversation exported to {file_path}")
    
    def export_knowledge(self):
        """Export knowledge base to file"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Knowledge",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            if self.deepnova.export_knowledge(file_path):
                self.add_message("system", f"Knowledge exported to {file_path}")
            else:
                self.add_message("system", "Failed to export knowledge")
    
    def cleanup_memory(self):
        """Clean up GPU memory"""
        cleanup_memory()
        self.add_message("system", "Memory cleanup performed")
        self.status_message.config(text="Memory cleaned")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
DeepNova AI Assistant v5.0

Professional MoE (Mixture of Experts) Transformer
with DeepSeek V3 architecture.

Features:
- MoE with top-k routing
- Parallel MoE + Dense MLP
- Adaptive Router with learnable temperature
- Dynamic Depth (layer skipping)
- Multi-Token Prediction (MTP)
- Intelligent context memory
- Persistent knowledge storage

Created with advanced AI technology.
"""
        messagebox.showinfo("About DeepNova", about_text)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        shortcuts_text = """
Keyboard Shortcuts:

Ctrl+Enter / Cmd+Enter - Send message
Ctrl+O - Load model
Ctrl+L - Clear chat

Slash Commands:
/clear - Clear conversation
/stats - Show statistics
/learn <text> - Learn new information
/recall <query> - Recall knowledge
"""
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = DeepNovaGUI(root)
    
    # Handle window close
    def on_closing():
        cleanup_memory()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()