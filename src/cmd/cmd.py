
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import os
import sys
import json
import time
import subprocess
import shlex
from datetime import datetime
from pathlib import Path

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import (
        DeepNovaAI, Transformer, ProductionTokenizer, ModelArgs,
        load_model, save_model, get_best_device, get_memory_info,
        cleanup_memory, logger, torch
    )
    MODEL_OK = True
except ImportError as e:
    print(f"Error: {e}")
    MODEL_OK = False
    sys.exit(1)


class DevTool:
    """Professional development tool with CMD support"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DeepNova DevTool - AI Development Environment")
        self.root.geometry("1400x950")
        self.root.configure(bg="#1e1e1e")
        
        # Model instances
        self.model = None
        self.tokenizer = None
        self.args = None
        self.deepnova = None
        self.model_loaded = False
        self.loading = False
        
        # CMD variables
        self.current_dir = os.getcwd()
        self.cmd_history = []
        self.history_index = 0
        self.cmd_process = None
        
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
        
        # Colors
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
            'info': '#9cdcfe',
            'cmd_prompt': '#4ec9b0',
            'cmd_output': '#d4d4d4',
            'cmd_error': '#f48771'
        }
        
        # Build UI
        self.create_menu()
        self.create_layout()
        self.setup_shortcuts()
        
        # Start monitoring
        self.start_monitoring()
        
        # Auto load
        self.root.after(1000, self.auto_load)
        
        # CMD welcome
        self.log_to_cmd("DeepNova CMD v1.0 initialized", "info")
        self.log_to_cmd(f"Current directory: {self.current_dir}", "info")
        self.log_to_cmd("Type 'help' for available commands", "info")
        self.log_to_cmd("Type 'exit' to close CMD", "info")
        self.print_cmd_prompt()
    
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
        model_menu.add_command(label="Run Benchmark", command=self.run_benchmark)
        model_menu.add_command(label="Profile Model", command=self.profile_model)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Code Generator", command=self.code_generator)
        tools_menu.add_command(label="Prompt Optimizer", command=self.prompt_optimizer)
        tools_menu.add_separator()
        tools_menu.add_command(label="Memory Cleanup", command=self.cleanup)
        
        # CMD menu
        cmd_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="CMD", menu=cmd_menu)
        cmd_menu.add_command(label="Clear CMD", command=self.clear_cmd)
        cmd_menu.add_command(label="Run Script", command=self.run_script)
        cmd_menu.add_separator()
        cmd_menu.add_command(label="Install Package", command=self.install_package)
        cmd_menu.add_command(label="List Files", command=self.list_files)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], fg=self.colors['fg'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="CMD Commands", command=self.show_cmd_help)
        help_menu.add_command(label="Documentation", command=self.show_docs)
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
        
        # Bottom: CMD Terminal
        self.create_cmd_terminal()
    
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
    
    def create_chat_tab(self, parent):
        """Create chat interface"""
        self.chat_display = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, font=("Consolas", 10),
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
        self.console.insert(tk.END, "Python Console (DeepNova)\n")
        self.console.insert(tk.END, ">>> ")
        self.console.mark_set("prompt", "insert-1c")
        self.console.mark_gravity("prompt", tk.LEFT)
        
        self.console.bind("<Control-Return>", self.execute_console)
        self.console.bind("<Key>", self.handle_console_key)
    
    def create_cmd_terminal(self):
        """Create CMD terminal at bottom"""
        self.cmd_frame = tk.Frame(self.root, bg=self.colors['terminal'], height=250)
        self.cmd_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.cmd_frame.pack_propagate(False)
        
        # CMD toolbar
        toolbar = tk.Frame(self.cmd_frame, bg=self.colors['sidebar'], height=30)
        toolbar.pack(fill=tk.X)
        toolbar.pack_propagate(False)
        
        tk.Label(toolbar, text="CMD Terminal", bg=self.colors['sidebar'],
                fg=self.colors['fg'], font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=10)
        
        tk.Label(toolbar, text=f"Dir: {self.current_dir}", bg=self.colors['sidebar'],
                fg=self.colors['info'], font=("Consolas", 8)).pack(side=tk.LEFT, padx=10)
        
        clear_cmd_btn = tk.Button(toolbar, text="Clear", command=self.clear_cmd,
                                  bg=self.colors['accent'], fg="white", height=1, padx=10)
        clear_cmd_btn.pack(side=tk.RIGHT, padx=5)
        
        # CMD display
        self.cmd_display = scrolledtext.ScrolledText(
            self.cmd_frame, wrap=tk.WORD, font=("Consolas", 10),
            bg=self.colors['terminal'], fg=self.colors['terminal_fg'],
            height=12
        )
        self.cmd_display.pack(fill=tk.BOTH, expand=True)
        self.cmd_display.config(state=tk.DISABLED)
        
        # CMD input
        input_frame = tk.Frame(self.cmd_frame, bg=self.colors['terminal'])
        input_frame.pack(fill=tk.X)
        
        self.cmd_prompt_label = tk.Label(input_frame, text=">", bg=self.colors['terminal'],
                                         fg=self.colors['cmd_prompt'], font=("Consolas", 10, "bold"))
        self.cmd_prompt_label.pack(side=tk.LEFT, padx=5)
        
        self.cmd_input = tk.Entry(input_frame, bg=self.colors['terminal'],
                                  fg=self.colors['terminal_fg'], font=("Consolas", 10),
                                  insertbackground=self.colors['accent'],
                                  relief="flat", highlightthickness=0)
        self.cmd_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.cmd_input.bind("<Return>", self.execute_cmd)
        self.cmd_input.bind("<Up>", self.cmd_history_up)
        self.cmd_input.bind("<Down>", self.cmd_history_down)
        
        # Configure text tags
        self.cmd_display.tag_config("cmd", foreground=self.colors['cmd_prompt'])
        self.cmd_display.tag_config("output", foreground=self.colors['cmd_output'])
        self.cmd_display.tag_config("error", foreground=self.colors['cmd_error'])
        self.cmd_display.tag_config("info", foreground=self.colors['info'])
        
        self.cmd_input.focus()
    
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
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind("<Control-o>", lambda e: self.load_model_dialog())
        self.root.bind("<Control-s>", lambda e: self.save_model())
        self.root.bind("<Control-q>", lambda e: self.root.quit())
        self.root.bind("<Control-l>", lambda e: self.clear_chat())
        self.input_field.bind("<Control-Return>", lambda e: self.send_message())
    
    # ============ CMD Functions ============
    
    def log_to_cmd(self, message, msg_type="output"):
        """Log message to CMD display"""
        self.cmd_display.config(state=tk.NORMAL)
        self.cmd_display.insert(tk.END, f"{message}\n", msg_type)
        self.cmd_display.see(tk.END)
        self.cmd_display.config(state=tk.DISABLED)
    
    def print_cmd_prompt(self):
        """Print command prompt"""
        self.cmd_display.config(state=tk.NORMAL)
        prompt = f"{self.current_dir}> "
        self.cmd_display.insert(tk.END, prompt, "cmd")
        self.cmd_display.see(tk.END)
        self.cmd_display.config(state=tk.DISABLED)
        self.cmd_input.delete(0, tk.END)
        self.cmd_input.focus()
    
    def execute_cmd(self, event=None):
        """Execute CMD command"""
        cmd = self.cmd_input.get().strip()
        if not cmd:
            self.print_cmd_prompt()
            return
        
        # Add to history
        self.cmd_history.append(cmd)
        self.history_index = len(self.cmd_history)
        
        # Log command
        self.cmd_display.config(state=tk.NORMAL)
        self.cmd_display.insert(tk.END, f"{self.current_dir}> {cmd}\n", "cmd")
        self.cmd_display.config(state=tk.DISABLED)
        
        # Handle built-in commands
        if cmd.lower() in ['exit', 'quit']:
            self.log_to_cmd("Exiting CMD...", "info")
            self.cmd_input.delete(0, tk.END)
            return
        
        elif cmd.lower() == 'clear' or cmd.lower() == 'cls':
            self.clear_cmd()
            self.print_cmd_prompt()
            return
        
        elif cmd.lower().startswith('cd '):
            self.change_directory(cmd[3:].strip())
            self.print_cmd_prompt()
            return
        
        elif cmd.lower() == 'dir' or cmd.lower() == 'ls':
            self.list_files()
            self.print_cmd_prompt()
            return
        
        elif cmd.lower() == 'pwd':
            self.log_to_cmd(self.current_dir, "output")
            self.print_cmd_prompt()
            return
        
        elif cmd.lower().startswith('python ') or cmd.lower().startswith('py '):
            self.run_python_script(cmd[cmd.find(' ')+1:])
            self.print_cmd_prompt()
            return
        
        elif cmd.lower().startswith('pip install '):
            self.install_package_cmd(cmd[12:])
            self.print_cmd_prompt()
            return
        
        elif cmd.lower() == 'help':
            self.show_cmd_help()
            self.print_cmd_prompt()
            return
        
        elif cmd.lower().startswith('model '):
            self.handle_model_cmd(cmd[6:].strip())
            self.print_cmd_prompt()
            return
        
        elif cmd.lower().startswith('chat '):
            self.handle_chat_cmd(cmd[5:].strip())
            return
        
        else:
            # Execute system command
            self.run_system_command(cmd)
            self.print_cmd_prompt()
    
    def run_system_command(self, cmd):
        """Run system command"""
        def run():
            try:
                process = subprocess.Popen(
                    cmd, shell=True, cwd=self.current_dir,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding='utf-8', errors='replace'
                )
                stdout, stderr = process.communicate(timeout=30)
                
                if stdout:
                    self.root.after(0, lambda: self.log_to_cmd(stdout.rstrip(), "output"))
                if stderr:
                    self.root.after(0, lambda: self.log_to_cmd(stderr.rstrip(), "error"))
                if process.returncode != 0:
                    self.root.after(0, lambda: self.log_to_cmd(f"Exit code: {process.returncode}", "error"))
                    
            except subprocess.TimeoutExpired:
                self.root.after(0, lambda: self.log_to_cmd("Command timeout (30s)", "error"))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_cmd(f"Error: {e}", "error"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def change_directory(self, path):
        """Change current directory"""
        try:
            if path == '..':
                new_path = os.path.dirname(self.current_dir)
            elif path == '.':
                new_path = self.current_dir
            elif os.path.isabs(path):
                new_path = path
            else:
                new_path = os.path.join(self.current_dir, path)
            
            if os.path.exists(new_path) and os.path.isdir(new_path):
                self.current_dir = os.path.abspath(new_path)
                self.log_to_cmd(f"Changed to: {self.current_dir}", "info")
                # Update toolbar
                for child in self.cmd_frame.winfo_children():
                    if isinstance(child, tk.Frame):
                        for widget in child.winfo_children():
                            if isinstance(widget, tk.Label) and 'Dir:' in widget.cget('text'):
                                widget.config(text=f"Dir: {self.current_dir}")
            else:
                self.log_to_cmd(f"Directory not found: {path}", "error")
        except Exception as e:
            self.log_to_cmd(f"Error: {e}", "error")
    
    def list_files(self):
        """List files in current directory"""
        try:
            files = os.listdir(self.current_dir)
            dirs = [f for f in files if os.path.isdir(os.path.join(self.current_dir, f))]
            files_only = [f for f in files if os.path.isfile(os.path.join(self.current_dir, f))]
            
            self.log_to_cmd(f"\nDirectory: {self.current_dir}", "info")
            self.log_to_cmd("-" * 50, "output")
            
            for d in sorted(dirs):
                self.log_to_cmd(f"[DIR]  {d}", "info")
            for f in sorted(files_only):
                size = os.path.getsize(os.path.join(self.current_dir, f))
                self.log_to_cmd(f"       {f} ({self.format_size(size)})", "output")
            
            self.log_to_cmd(f"\n{len(dirs)} dir(s), {len(files_only)} file(s)", "info")
        except Exception as e:
            self.log_to_cmd(f"Error: {e}", "error")
    
    def format_size(self, size):
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def run_python_script(self, script_path):
        """Run Python script"""
        if not os.path.exists(script_path):
            self.log_to_cmd(f"File not found: {script_path}", "error")
            return
        
        def run():
            try:
                result = subprocess.run(
                    ['python', script_path], cwd=self.current_dir,
                    capture_output=True, text=True, timeout=30
                )
                if result.stdout:
                    self.root.after(0, lambda: self.log_to_cmd(result.stdout.rstrip(), "output"))
                if result.stderr:
                    self.root.after(0, lambda: self.log_to_cmd(result.stderr.rstrip(), "error"))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_cmd(f"Error: {e}", "error"))
        
        threading.Thread(target=run, daemon=True).start()
    
    def install_package_cmd(self, package):
        """Install Python package via pip"""
        self.log_to_cmd(f"Installing {package}...", "info")
        
        def install():
            try:
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    self.root.after(0, lambda: self.log_to_cmd(f"Successfully installed {package}", "success"))
                else:
                    self.root.after(0, lambda: self.log_to_cmd(f"Failed: {result.stderr}", "error"))
            except Exception as e:
                self.root.after(0, lambda: self.log_to_cmd(f"Error: {e}", "error"))
        
        threading.Thread(target=install, daemon=True).start()
    
    def handle_model_cmd(self, cmd):
        """Handle model-related commands"""
        if not self.model_loaded:
            self.log_to_cmd("Model not loaded. Use 'model load' first", "error")
            return
        
        if cmd == 'info':
            info = self.deepnova.get_model_info()
            self.log_to_cmd(f"Model: {info.get('model_name', 'N/A')}", "info")
            self.log_to_cmd(f"Params: {info.get('total_params_formatted', 'N/A')}", "info")
            self.log_to_cmd(f"Active: {info.get('active_params_formatted', 'N/A')}", "info")
            self.log_to_cmd(f"Layers: {info.get('n_layers', 'N/A')}", "info")
            self.log_to_cmd(f"Experts: {info.get('n_experts', 'N/A')}", "info")
        elif cmd == 'stats':
            stats = self.deepnova.get_stats()
            self.log_to_cmd(f"Messages: {stats.get('total_messages', 0)}", "info")
            self.log_to_cmd(f"Tokens: {stats.get('total_tokens_generated', 0)}", "info")
            self.log_to_cmd(f"Learned: {stats.get('learning', {}).get('total_learned', 0)}", "info")
        elif cmd == 'clear':
            self.deepnova.clear_context(keep_important=True)
            self.log_to_cmd("Model memory cleared", "success")
        else:
            self.log_to_cmd(f"Unknown model command: {cmd}", "error")
            self.log_to_cmd("Available: info, stats, clear", "info")
    
    def handle_chat_cmd(self, message):
        """Handle chat command from CMD"""
        if not self.model_loaded:
            self.log_to_cmd("Model not loaded", "error")
            self.print_cmd_prompt()
            return
        
        self.log_to_cmd(f"User: {message}", "cmd")
        
        def chat():
            try:
                response = self.deepnova.chat(message, max_new_tokens=200, temperature=0.7)
                self.root.after(0, lambda: self.log_to_cmd(f"DeepNova: {response}", "output"))
                self.root.after(0, self.print_cmd_prompt)
            except Exception as e:
                self.root.after(0, lambda: self.log_to_cmd(f"Error: {e}", "error"))
                self.root.after(0, self.print_cmd_prompt)
        
        threading.Thread(target=chat, daemon=True).start()
    
    def cmd_history_up(self, event):
        """Navigate command history up"""
        if self.history_index > 0:
            self.history_index -= 1
            self.cmd_input.delete(0, tk.END)
            self.cmd_input.insert(0, self.cmd_history[self.history_index])
    
    def cmd_history_down(self, event):
        """Navigate command history down"""
        if self.history_index < len(self.cmd_history) - 1:
            self.history_index += 1
            self.cmd_input.delete(0, tk.END)
            self.cmd_input.insert(0, self.cmd_history[self.history_index])
        elif self.history_index == len(self.cmd_history) - 1:
            self.history_index = len(self.cmd_history)
            self.cmd_input.delete(0, tk.END)
    
    def clear_cmd(self):
        """Clear CMD display"""
        self.cmd_display.config(state=tk.NORMAL)
        self.cmd_display.delete(1.0, tk.END)
        self.cmd_display.config(state=tk.DISABLED)
    
    def run_script(self):
        """Run script file"""
        path = filedialog.askopenfilename(
            title="Select Script",
            filetypes=[("Python files", "*.py"), ("Batch files", "*.bat *.cmd"), ("All", "*.*")]
        )
        if path:
            self.cmd_input.delete(0, tk.END)
            if path.endswith('.py'):
                self.cmd_input.insert(0, f'python "{path}"')
            else:
                self.cmd_input.insert(0, f'"{path}"')
            self.execute_cmd()
    
    def install_package(self):
        """Install Python package dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Install Package")
        dialog.geometry("400x150")
        dialog.configure(bg=self.colors['bg'])
        
        tk.Label(dialog, text="Package name:", bg=self.colors['bg'],
                fg=self.colors['fg']).pack(pady=10)
        
        entry = tk.Entry(dialog, width=30, font=("Consolas", 10))
        entry.pack(pady=5)
        
        def install():
            pkg = entry.get().strip()
            if pkg:
                dialog.destroy()
                self.cmd_input.delete(0, tk.END)
                self.cmd_input.insert(0, f'pip install {pkg}')
                self.execute_cmd()
        
        tk.Button(dialog, text="Install", command=install,
                 bg=self.colors['accent'], fg="white", padx=20).pack(pady=10)
    
    def show_cmd_help(self):
        """Show CMD help"""
        help_text = """
DeepNova CMD Commands:
======================

Navigation:
  cd <path>     - Change directory
  dir / ls      - List files in current directory
  pwd           - Show current directory
  clear / cls   - Clear CMD screen

File Operations:
  python <file> - Run Python script
  py <file>     - Run Python script
  
Package Management:
  pip install <package> - Install Python package

Model Commands (requires loaded model):
  model info    - Show model information
  model stats   - Show model statistics  
  model clear   - Clear model memory
  chat <text>   - Chat with AI

System:
  exit / quit   - Exit CMD
  help          - Show this help

Examples:
  cd Documents
  dir
  python script.py
  pip install numpy
  model info
  chat Hello, how are you?
"""
        self.log_to_cmd(help_text, "info")
    
    # ============ Existing Functions ============
    
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
        self.log_to_cmd("Loading model...", "info")
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
        self.log_to_cmd("Model loaded successfully!", "success")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
        
        info = self.deepnova.get_model_info()
        self.model_info.delete(1.0, tk.END)
        self.model_info.insert(tk.END, f"Params: {info.get('total_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Active: {info.get('active_params_formatted', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Layers: {info.get('n_layers', 'N/A')}\n")
        self.model_info.insert(tk.END, f"Experts: {info.get('n_experts', 'N/A')}")
    
    def on_load_error(self, error):
        """Handle load error"""
        self.log_to_cmd(f"Load failed: {error}", "error")
        self.load_btn.config(state=tk.NORMAL, text="Load Model")
    
    def send_message(self):
        """Send chat message"""
        if not self.model_loaded:
            self.log_to_cmd("Model not loaded", "error")
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
        self.log_to_cmd("Generation stopped", "warning")
    
    def add_chat_message(self, role, msg):
        """Add message to chat"""
        self.chat_display.config(state=tk.NORMAL)
        time_str = datetime.now().strftime("%H:%M:%S")
        self.chat_display.insert(tk.END, f"\n[{time_str}] {role.upper()}:\n", role)
        self.chat_display.insert(tk.END, f"{msg}\n")
        self.chat_display.insert(tk.END, "-" * 60 + "\n")
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def clear_chat(self):
        """Clear chat display"""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        self.log_to_cmd("Chat cleared", "info")
    
    def run_code(self):
        """Run code from editor"""
        code = self.code_editor.get(1.0, tk.END)
        self.log_to_cmd("Executing code...", "info")
        
        def execute():
            try:
                exec(code, {"__name__": "__main__", "torch": torch, "model": self.model})
                self.log_to_cmd("Code executed successfully", "success")
            except Exception as e:
                self.log_to_cmd(f"Code error: {e}", "error")
        
        threading.Thread(target=execute, daemon=True).start()
    
    def analyze_code(self):
        """Analyze code"""
        code = self.code_editor.get(1.0, tk.END)
        lines = len(code.split('\n'))
        chars = len(code)
        self.log_to_cmd(f"Code analysis: {lines} lines, {chars} characters", "info")
    
    def format_code(self):
        """Format code (basic)"""
        code = self.code_editor.get(1.0, tk.END)
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
        self.log_to_cmd("Code formatted", "info")
    
    def execute_console(self, event=None):
        """Execute Python console command"""
        code = self.console.get("prompt", tk.END).strip()
        if not code:
            return
        
        self.console.insert(tk.END, "\n")
        
        def execute():
            try:
                result = eval(code, {"__builtins__": __builtins__, "torch": torch, "model": self.model})
                if result is not None:
                    self.console.insert(tk.END, f"{result}\n")
            except:
                try:
                    exec(code, {"__builtins__": __builtins__, "torch": torch, "model": self.model})
                except Exception as e:
                    self.console.insert(tk.END, f"Error: {e}\n")
            
            self.console.insert(tk.END, ">>> ")
            self.console.mark_set("prompt", "insert-1c")
            self.console.see(tk.END)
        
        threading.Thread(target=execute, daemon=True).start()
    
    def handle_console_key(self, event):
        """Handle console key presses"""
        if event.keysym == "Return" and not event.state & 0x4:  # Ctrl not pressed
            self.execute_console()
            return "break"
    
    def run_benchmark(self):
        """Run comprehensive benchmark"""
        self.log_to_cmd("Running benchmark...", "info")
        if self.model:
            dummy = torch.randint(0, 1000, (1, 32))
            times = []
            for i in range(10):
                start = time.time()
                _ = self.model(dummy)
                times.append(time.time() - start)
            avg = sum(times) / len(times)
            self.log_to_cmd(f"Average forward time: {avg*1000:.2f}ms", "info")
    
    def profile_model(self):
        """Profile model"""
        self.log_to_cmd("Profiling model...", "info")
        if self.model:
            total = sum(p.numel() for p in self.model.parameters())
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.log_to_cmd(f"Total params: {total:,}", "info")
            self.log_to_cmd(f"Trainable: {trainable:,}", "info")
    
    def show_model_info(self):
        """Show model information"""
        if self.deepnova:
            info = self.deepnova.get_model_info()
            msg = json.dumps(info, indent=2)
            messagebox.showinfo("Model Info", msg)
    
    def show_params(self):
        """Show parameter count"""
        if self.model:
            total = sum(p.numel() for p in self.model.parameters())
            messagebox.showinfo("Parameters", f"Total parameters: {total:,}")
    
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
        self.log_to_cmd("Code template generated", "info")
    
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
            optimized = f"System: You are a helpful assistant.\nUser: {prompt}\nAssistant:"
            result_text.delete(1.0, tk.END)
            result_text.insert(1.0, optimized)
        
        tk.Button(dialog, text="Optimize", command=optimize,
                 bg=self.colors['accent'], fg="white", padx=20).pack(pady=10)
    
    def cleanup(self):
        """Clean up memory"""
        cleanup_memory()
        self.log_to_cmd("Memory cleaned", "success")
    
    def save_model(self):
        """Save model"""
        if not self.model_loaded:
            self.log_to_cmd("No model to save", "error")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pt")
        if path:
            save_model(self.model, os.path.dirname(path))
            self.log_to_cmd(f"Model saved to {path}", "success")
    
    def export_model(self):
        """Export model"""
        self.save_model()
    
    def show_docs(self):
        """Show documentation"""
        doc_text = """DeepNova DevTool Documentation

Features:
- Model loading and management
- Interactive chat with AI
- Code editor with Python support
- Python console for testing
- CMD terminal with system commands
- Performance benchmarking
- Memory profiling

CMD Commands:
- cd, dir, pwd, clear
- python <file>, pip install
- model info, model stats
- chat <text>

For more information, visit documentation."""
        messagebox.showinfo("Documentation", doc_text)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """DeepNova DevTool v2.0

Professional development environment for DeepNova AI

Features:
- MoE Transformer architecture
- Multi-expert routing
- Intelligent memory system
- Code generation tools
- CMD terminal support

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


if __name__ == "__main__":
    app = DevTool()
    app.run()