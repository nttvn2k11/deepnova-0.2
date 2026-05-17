# DeepNova AI

![DeepNova](https://img.shields.io/badge/DeepNova-AI%20Assistant-blue)
![Python](https://img.shields.io/badge/Python-3.9+-brightgreen)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)

A professional, multi-interface AI assistant with advanced MoE (Mixture of Experts) + Dense + GLM Transformer architecture. Provides desktop GUI, web, and command-line interfaces for seamless interaction with state-of-the-art language models.

**License:** Apache 2.0

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Detailed Python Setup Guide](#detailed-python-setup-guide)
  - [Prerequisites Check](#prerequisites-check)
  - [System Setup Workflow](#system-setup-workflow)
  - [Step 1: Download Python](#step-1-download-python)
  - [Step 2: Verify Python Installation](#step-2-verify-python-installation)
  - [Step 3: Create Virtual Environment](#step-3-create-virtual-environment)
  - [Step 4: Activate Virtual Environment](#step-4-activate-virtual-environment)
  - [Step 5: Deactivate Virtual Environment](#step-5-deactivate-virtual-environment)
  - [Step 6: Install Dependencies](#step-6-install-dependencies---full-guide)
  - [Step 7: Verify Installation](#step-7-verify-installation-complete)
  - [Troubleshooting](#python-installation-troubleshooting)
  - [Requirements File Management](#requirements-file-management)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Desktop GUI](#desktop-gui)
  - [Web Interface](#web-interface)
  - [Development Tool](#development-tool)
  - [Python API](#python-api)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Multiple Interfaces:**
  - Professional Desktop GUI (Tkinter)
  - Web Interface with REST API (Flask)
  - Command-Line Development Tool
  - Simple Chat Application
  
- **Advanced Architecture:**
  - Mixture of Experts (MoE) for efficient scaling
  - Dense layers for context understanding
  - GLM Transformer base model
  - Support for multiple model sizes (lite, standard, large)

- **Cross-Platform:**
  - Linux (Ubuntu 20.04+)
  - macOS 12+
  - Windows 10/11

- **GPU Acceleration:**
  - NVIDIA CUDA support
  - Apple Metal Performance Shaders (MPS)
  - Intel XPU support
  - Fallback to CPU

- **Professional Features:**
  - Comprehensive logging system
  - Memory management and optimization
  - Model saving/loading
  - Configurable generation parameters
  - Performance monitoring
  - API endpoints for integration

---

## Project Structure

```
cat-ai/
│
├── readme.md                           # Project documentation (this file)
│
├── src/                                # Main source code directory
│   │
│   ├── app/
│   │   └── app.py                     # Desktop GUI application (Tkinter)
│   │                                  # - White interface with blue buttons
│   │                                  # - Real-time chat interface
│   │                                  # - Settings panel for config
│   │                                  # - Memory monitoring
│   │
│   ├── cmd/
│   │   └── cmd.py                     # Development tool with terminal
│   │                                  # - CMD/Terminal integration
│   │                                  # - Model management
│   │                                  # - Benchmarking tools
│   │                                  # - Code execution support
│   │
│   ├── gui/
│   │   ├── gui_chat.py                # Simple chat interface (Tkinter)
│   │   │                              # - Lightweight, minimal design
│   │   │                              # - Quick interactions
│   │   │                              # - Easy to customize
│   │   │
│   │   └── gui_setting.py             # VSCode-style GUI (Tkinter)
│   │                                  # - Professional IDE interface
│   │                                  # - Advanced settings
│   │                                  # - Sidebar with options
│   │
│   ├── model/
│   │   └── model.py                   # Core AI model implementation
│   │                                  # - DeepNovaAI class
│   │                                  # - Transformer architecture
│   │                                  # - MoE + Dense + GLM layers
│   │                                  # - Model loading/saving
│   │                                  # - Memory management
│   │                                  # - Device detection (CUDA/MPS/CPU)
│   │                                  # - Tokenizer integration
│   │                                  # - Inference functions
│   │                                  # - Logger setup
│   │
│   ├── tool/
│   │   └── code.py                    # Professional development tool (Tkinter)
│   │                                  # - Dark theme editor
│   │                                  # - Code execution
│   │                                  # - Terminal integration
│   │                                  # - File browser
│   │                                  # - Debugging support
│   │                                  # - Performance monitoring
│   │
│   ├── web/
│   │   └── web.py                     # Flask web server
│   │                                  # - REST API endpoints
│   │                                  # - Web-based chat interface
│   │                                  # - HTML template
│   │                                  # - JSON request/response
│   │                                  # - CORS support
│   │                                  # - Model management endpoints
│   │
│   └── logs/                          # Application logs directory
│       └── deepnova.log               # Main log file (auto-created)
│
├── checkpoints/                       # Model checkpoints (optional)
│   └── model_lite.pth                 # Saved model weights
│
├── requirements.txt                   # Python dependencies list
│   │                                  # - PyTorch
│   │                                  # - Transformers
│   │                                  # - Flask
│   │                                  # - All utilities
│
├── .gitignore                         # Git ignore rules (optional)
│   │                                  # - logs/
│   │                                  # - venv/
│   │                                  # - __pycache__/
│   │                                  # - *.pyc
│
└── config.yaml                        # Configuration file (optional)
                                       # - Model settings
                                       # - Training parameters
                                       # - Device configuration
```

### File Descriptions

#### Core Files

| File | Purpose | Key Components |
|------|---------|-----------------|
| **model.py** | Main AI model implementation | `DeepNovaAI`, `Transformer`, `ProductionTokenizer`, `load_model()`, `get_best_device()` |
| **app.py** | Desktop GUI application | Main entry point for desktop users |
| **gui_chat.py** | Simple chat interface | Lightweight alternative GUI |
| **gui_setting.py** | VSCode-style GUI | Professional interface with advanced settings |
| **cmd.py** | Development tool | Developer-focused terminal interface |
| **web.py** | Flask web server | REST API and web interface |

#### Directory Structure Explanation

**src/app/** - Desktop Application
- User-friendly Tkinter interface
- White background, blue buttons
- Settings panel for parameters
- Memory and performance display

**src/cmd/** - Command-Line Tool
- Development environment
- Terminal/CMD integration
- Benchmarking capabilities
- Model debugging tools

**src/gui/** - Alternative GUIs
- `gui_chat.py`: Simple, fast interface
- `gui_setting.py`: Professional IDE-style interface

**src/model/** - Core AI Engine
- Deep learning model implementation
- All AI/ML computation
- Device management (GPU/CPU)
- Tokenization and inference

**src/tool/** - Advanced Tools
- Code execution
- File browsing
- Performance analysis
- Debugging utilities

**src/web/** - Web Interface
- REST API server
- HTML web interface
- Remote access capability
- JSON data handling

**src/logs/** - Application Logging
- Auto-created log directory
- Debug and error logs
- Performance metrics logging

### Launch Files

```
Quick Start Commands:

Desktop GUI:
    python src/app/app.py

Simple Chat:
    python src/gui/gui_chat.py

VSCode GUI:
    python src/gui/gui_setting.py

Web Server:
    python src/web/web.py

Dev Tool:
    python src/cmd/cmd.py
```

### File Dependencies

```
Dependency Graph:
    
    All GUIs & Tools → model.py (core)
                    ↓
                PyTorch
                Transformers
                NumPy
                
    web.py → Flask/Flask-CORS
           ↓
           model.py
           
    app.py → Tkinter (built-in)
           ↓
           model.py
           
    gui_chat.py → Tkinter (built-in)
                ↓
                model.py
                
    gui_setting.py → Tkinter (built-in)
                   ↓
                   model.py
                   
    cmd.py → Tkinter (built-in)
           ↓
           model.py
```

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Linux (Ubuntu 20.04+), macOS 12+, Windows 10/11 |
| **Python** | 3.9 or higher |
| **CPU** | 4 cores, 2.5 GHz |
| **RAM** | 16 GB |
| **Storage** | 30 GB free space |
| **GPU** | Optional: NVIDIA GPU with 8GB+ VRAM |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **OS** | Ubuntu 22.04 LTS / macOS 13+ / Windows 11 |
| **Python** | 3.10 or higher |
| **CPU** | 16 cores, 3.0 GHz+ |
| **RAM** | 64 GB+ |
| **Storage** | 100 GB SSD |
| **GPU** | NVIDIA RTX 3090/4090, A100, or H100 (24GB+ VRAM) |

---

## Detailed Python Setup Guide

This section provides comprehensive step-by-step instructions for setting up Python from scratch.

### Prerequisites Check

Before starting, verify your system:

```bash
# Check Python version
python --version
# or
python3 --version

# Check pip is installed
pip --version
# or
pip3 --version

# Check system architecture (32-bit or 64-bit)
python -c "import struct; print(struct.calcsize('P') * 8, 'bit')"
```

### System Setup Workflow

```
Start
  |
  +---> Step 1: Download Python
  |       └---> Select correct version for your OS
  |
  +---> Step 2: Install Python
  |       └---> Enable "Add Python to PATH" (Windows)
  |
  +---> Step 3: Verify Installation
  |       └---> Check python --version
  |
  +---> Step 4: Create Virtual Environment
  |       └---> Keep projects isolated
  |
  +---> Step 5: Activate Virtual Environment
  |       └---> Select active Python environment
  |
  +---> Step 6: Install Dependencies
  |       └---> pip install -r requirements.txt
  |
  +---> Step 7: Verify All Packages
  |       └---> Import test all modules
  |
  End
```

### Step 1: Download Python

#### Windows

1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.10" (or latest 3.9+)
3. Run the installer
4. **IMPORTANT:** Check "Add Python to PATH"
5. Choose "Install Now" or "Customize Installation"

```
Installer Screen:
┌─────────────────────────────────────────────┐
│  Python 3.10 Setup                          │
├─────────────────────────────────────────────┤
│  [✓] Install launcher for all users         │
│  [✓] Add Python 3.10 to PATH  <-- CHECK!   │
│                                              │
│  [Install Now]  [Customize Installation]   │
└─────────────────────────────────────────────┘
```

#### macOS

```bash
# Option 1: Using Homebrew (recommended)
brew install python3

# Option 2: Download from python.org
# Visit https://www.python.org/downloads/mac-osx/
# Download and run the installer
```

#### Linux (Ubuntu/Debian)

```bash
# Update package manager
sudo apt update
sudo apt upgrade

# Install Python 3
sudo apt install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

#### Linux (Fedora/RHEL/CentOS)

```bash
# Update package manager
sudo dnf update

# Install Python 3
sudo dnf install python3 python3-pip

# Verify installation
python3 --version
pip3 --version
```

### Step 2: Verify Python Installation

```bash
# Check Python executable location
which python3          # macOS/Linux
where python           # Windows

# Check Python version
python --version       # Should show Python 3.9+

# Check pip version
pip --version          # Should show pip 20.0+

# Check installed Python modules
python -c "import sys; print('\n'.join(sys.path))"

# Check if tkinter is available
python -c "import tkinter; print('Tkinter available')"
```

### Step 3: Create Virtual Environment

Virtual environments keep your project dependencies isolated from system Python.

```bash
# Navigate to project directory
cd deepnova-ai

# Create virtual environment
python -m venv venv

# Directory structure after creation:
# venv/
# ├── bin/              (macOS/Linux)
# │   ├── python
# │   ├── pip
# │   ├── activate
# │   └── ...
# ├── Scripts/          (Windows)
# │   ├── python.exe
# │   ├── pip.exe
# │   ├── activate.bat
# │   └── ...
# ├── lib/
# ├── include/
# └── pyvenv.cfg
```

### Step 4: Activate Virtual Environment

#### Windows (PowerShell)

```powershell
# Navigate to project
cd C:\Users\YourName\Desktop\deepnova-ai

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# You should see: (venv) in your prompt
# (venv) PS C:\Users\YourName\Desktop\deepnova-ai>
```

#### Windows (Command Prompt)

```cmd
# Navigate to project
cd C:\Users\YourName\Desktop\deepnova-ai

# Activate virtual environment
venv\Scripts\activate.bat

# You should see: (venv) in your prompt
# (venv) C:\Users\YourName\Desktop\deepnova-ai>
```

#### macOS/Linux

```bash
# Navigate to project
cd ~/Desktop/deepnova-ai

# Activate virtual environment
source venv/bin/activate

# You should see: (venv) in your prompt
# (venv) ~/Desktop/deepnova-ai $
```

**Verify activation:**

```bash
# Check which Python is active
which python              # macOS/Linux - should show path with venv
where python              # Windows

# Check Python location in venv
python -c "import sys; print(sys.executable)"
# Output should show venv path

# Test pip
pip --version
# Output should reference venv
```

### Step 5: Deactivate Virtual Environment

```bash
# Any OS
deactivate

# Prompt returns to normal (no (venv) prefix)
```

### Step 6: Install Dependencies - Full Guide

#### Upgrade pip, setuptools, and wheel

```bash
# Verify virtual environment is activated
# (should see (venv) in prompt)

# Upgrade pip
pip install --upgrade pip

# Upgrade setuptools and wheel
pip install --upgrade setuptools wheel
```

#### Install Core Dependencies

```bash
# PyTorch with CPU support (lightweight)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# OR PyTorch with GPU support (NVIDIA CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# OR PyTorch with GPU support (NVIDIA CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Which PyTorch version?**

```
Check your NVIDIA GPU driver:
  nvidia-smi

Version mapping:
  ┌──────────────────────────────────────────┐
  │ CUDA Version → PyTorch Index             │
  ├──────────────────────────────────────────┤
  │ No GPU (CPU only) → cu121 or cpu        │
  │ CUDA 11.8+        → cu118 or cu121      │
  │ CUDA 12.1+        → cu121                │
  │ Apple Silicon     → default (has MPS)    │
  │ AMD GPU           → rocm                 │
  └──────────────────────────────────────────┘
```

#### Install Transformers and Tokenizers

```bash
pip install transformers>=4.36.0
pip install sentencepiece>=0.1.99
pip install tokenizers>=0.15.0
```

#### Install Core Utilities

```bash
pip install numpy>=1.24.0
pip install tqdm>=4.66.0
pip install psutil>=5.9.0
pip install safetensors>=0.4.0
```

#### Install Web Framework

```bash
pip install flask>=2.3.0
pip install flask-cors>=4.0.0
```

#### Install Optional Performance Packages

```bash
# Flash Attention (FASTER inference)
pip install flash-attn>=2.3.0 --no-build-isolation

# Triton (Custom CUDA kernels)
pip install triton>=2.1.0

# Megablocks (MoE optimization)
pip install megablocks>=0.6.0
```

### Step 7: Verify Installation Complete

#### Check All Modules Import Successfully

```bash
# Create test file: test_imports.py
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""Test all required modules"""

import sys
print(f"Python Version: {sys.version}")
print(f"Python Executable: {sys.executable}\n")

modules_to_test = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('tokenizers', 'Tokenizers'),
    ('numpy', 'NumPy'),
    ('tqdm', 'tqdm'),
    ('psutil', 'psutil'),
    ('safetensors', 'safetensors'),
    ('flask', 'Flask'),
    ('flask_cors', 'Flask-CORS'),
    ('tkinter', 'Tkinter'),
]

print("Testing module imports:")
print("-" * 50)

all_ok = True
for module_name, display_name in modules_to_test:
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"[OK] {display_name:20} v{version}")
    except ImportError as e:
        print(f"[FAIL] {display_name:20} - {e}")
        all_ok = False

print("-" * 50)
if all_ok:
    print("\nAll modules imported successfully!")
else:
    print("\nSome modules failed. Please install missing packages.")
    sys.exit(1)

# Check PyTorch device
try:
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nPyTorch device: {device}")
    if device == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except Exception as e:
    print(f"Error checking device: {e}")
EOF

# Run test
python test_imports.py
```

#### Expected Output

```
Python Version: 3.10.12 (main, Jun ...)
Python Executable: /path/to/venv/bin/python

Testing module imports:
--------------------------------------------------
[OK] PyTorch             v2.1.0
[OK] Transformers       v4.36.0
[OK] Tokenizers         v0.15.0
[OK] NumPy              v1.24.0
[OK] tqdm               v4.66.0
[OK] psutil             v5.9.0
[OK] safetensors        v0.4.0
[OK] Flask              v2.3.0
[OK] Flask-CORS         v4.0.0
[OK] Tkinter            N/A
--------------------------------------------------

All modules imported successfully!

PyTorch device: cuda
GPU Name: NVIDIA GeForce RTX 3090
GPU Memory: 24.0 GB
```

### Python Installation Troubleshooting

#### Problem: "python: command not found"

**Windows:**
```
Solution 1: Add Python to PATH
- Windows Settings → Environment Variables
- Add: C:\Users\YourName\AppData\Local\Programs\Python\Python310
- Restart terminal

Solution 2: Use full path
py --version
python.exe --version
```

**macOS/Linux:**
```bash
# Solution 1: Install Python
brew install python3  # macOS
sudo apt install python3  # Ubuntu

# Solution 2: Use python3 instead of python
python3 --version
alias python=python3
```

#### Problem: "pip: command not found"

```bash
# Reinstall pip
python -m pip install --upgrade pip

# Or use:
python -m pip install package_name
```

#### Problem: Virtual Environment Won't Activate

```bash
# Windows PowerShell: Execution Policy Error
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Linux/macOS: Permission denied
chmod +x venv/bin/activate
source venv/bin/activate
```

#### Problem: Module Import Fails

```bash
# Check if package installed
pip list | grep package_name

# Reinstall package
pip uninstall package_name
pip install package_name

# Check for conflicts
pip check
```

#### Problem: GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Requirements File Management

#### Generate requirements.txt

```bash
# Save all installed packages
pip freeze > requirements.txt

# View requirements
cat requirements.txt
```

#### Install from requirements.txt

```bash
# Install all packages listed
pip install -r requirements.txt

# Install with specific version constraints
pip install -r requirements.txt --no-deps
```

#### Example requirements.txt

```
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
transformers==4.36.0
sentencepiece==0.1.99
tokenizers==0.15.0
numpy==1.24.0
tqdm==4.66.0
psutil==5.9.0
safetensors==0.4.0
flask==2.3.0
flask-cors==4.0.0
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/deepnova-ai.git
cd deepnova-ai
```

### Step 2: Create Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install PyTorch (with CUDA support for GPU)
pip install torch>=2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install transformers>=4.36.0 sentencepiece>=0.1.99 tokenizers>=0.15.0
pip install numpy>=1.24.0 tqdm>=4.66.0 psutil>=5.9.0 safetensors>=0.4.0

# Install GUI dependencies
pip install flask flask-cors

# Optional: Enhanced performance packages
# pip install flash-attn>=2.3.0 --no-build-isolation
# pip install triton>=2.1.0
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Quick Start

### 1. Desktop GUI Application

```bash
python src/app/app.py
```

- Modern white interface with blue buttons
- Real-time chat with the AI model
- Settings panel for configuration
- Memory and performance monitoring

### 2. Simple Chat Interface

```bash
python src/gui/gui_chat.py
```

- Lightweight chat application
- Clean, minimal interface
- Perfect for quick interactions

### 3. VSCode-Style GUI

```bash
python src/gui/gui_setting.py
```

- Professional IDE-like interface
- Advanced configuration options
- Sidebar with detailed settings

### 4. Web Interface

```bash
python src/web/web.py
```

Server starts at `http://localhost:5000`

- REST API endpoints
- Web-based chat interface
- JSON request/response format

### 5. Development Tool

```bash
python src/cmd/cmd.py
```

- Terminal integration
- Developer-focused interface
- Benchmarking capabilities
- Command execution support

---

## Usage

### Desktop GUI

1. Run the application:
   ```bash
   python src/app/app.py
   ```

2. Wait for the model to load (first time may take a few minutes)

3. Type your message in the input field

4. Click "Send" or press `Enter`

5. Configure settings in the sidebar:
   - Temperature (0.0 - 1.0)
   - Max tokens
   - Model size

### Web Interface

**Start the server:**
```bash
python src/web/web.py
```

**API Endpoints:**

```bash
# Health check
curl http://localhost:5000/api/health

# Load model
curl -X POST http://localhost:5000/api/model/load

# Generate response
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'

# Chat conversation
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is AI?", "temperature": 0.7}'
```

### Python API

```python
from src.model.model import DeepNovaAI, load_model, get_best_device

# Initialize
device = get_best_device()
model = load_model("lite", device)
deepnova = DeepNovaAI(model, device)

# Generate text
prompt = "What is artificial intelligence?"
response = deepnova.generate(
    prompt,
    max_tokens=200,
    temperature=0.7,
    top_p=0.9
)
print(response)

# Clean up
deepnova.cleanup()
```

---

## Architecture

### Model Architecture

**DeepNova** uses a hybrid transformer architecture:

```
Input → Tokenizer
        ↓
    [GLM Base Transformer]
        ↓
    [Dense Layers]
        ↓
    [Mixture of Experts (MoE)]
        ↓
    [Output Head]
        ↓
    Generated Text
```

### Key Components

1. **GLM Transformer Base:** Foundation model for language understanding
2. **Dense Layers:** For feature extraction and context processing
3. **Mixture of Experts:** For efficient, scalable computation
4. **Production Tokenizer:** Optimized text tokenization
5. **Memory Manager:** Efficient GPU/CPU memory utilization

### Device Support

- **CUDA** (NVIDIA GPUs): Fastest option
- **MPS** (Apple Silicon): Native acceleration on Mac
- **XPU** (Intel Arc): Hardware-accelerated inference
- **NPU** (Neural Processing Unit): Specialized accelerators
- **CPU** (Fallback): Works on any machine

---

## Configuration

### Environment Variables

```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs

# Memory Settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Logging
export LOG_LEVEL=INFO
```

### Model Configuration

Edit parameters in code or via GUI:

```python
ModelArgs(
    dim=512,                    # Hidden dimension
    n_heads=8,                  # Number of attention heads
    n_layers=12,                # Number of transformer layers
    n_kv_heads=8,              # KV cache heads
    vocab_size=32000,          # Vocabulary size
    ff_dim=2048,               # Feed-forward dimension
    norm_eps=1e-5,             # Layer norm epsilon
    rope_theta=10000.0,        # RoPE theta
    moe_num_experts=8,         # Number of MoE experts
    moe_top_k=2,               # MoE top-k routing
)
```

### Generation Parameters

```python
# Temperature (lower = more deterministic)
temperature = 0.7

# Top-p sampling (nucleus sampling)
top_p = 0.9

# Top-k sampling
top_k = 50

# Repetition penalty
repeat_penalty = 1.0

# Maximum tokens to generate
max_tokens = 500
```

---

## Performance Tuning

### GPU Optimization

```python
# Enable mixed precision
torch.set_float32_matmul_precision('medium')

# Use CUDA graphs for faster inference
torch.cuda.empty_cache()

# Batch processing
batch_size = 8
```

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller batch sizes
batch_size = 1

# Monitor memory
from model import get_memory_info
memory = get_memory_info()
print(f"Memory usage: {memory}")
```

### Inference Speed

- **Batch Processing:** Process multiple queries simultaneously
- **Model Quantization:** Reduce model size and improve speed
- **KV Cache:** Reuse computed keys/values
- **Compile Mode:** Use `torch.compile()` for optimization

---

## Troubleshooting

### CUDA Not Available

```python
# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# Install correct PyTorch version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory (OOM)

```python
# Reduce batch size
batch_size = 1

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()
```

### Model Not Loading

```bash
# Check logs
cat logs/deepnova.log

# Verify file permissions
ls -la src/model/

# Check available disk space
df -h
```

### GUI Not Starting

```bash
# Install tkinter (Debian/Ubuntu)
sudo apt-get install python3-tk

# For macOS
brew install python-tk

# For Windows - included with Python installation
```

---

## Performance Metrics

| Metric | Lite | Standard | Large |
|--------|------|----------|-------|
| Model Size | ~3GB | ~7GB | ~13GB |
| Inference Speed (GPU) | 50 tokens/s | 30 tokens/s | 15 tokens/s |
| Memory Required | 8GB | 16GB | 24GB |
| Max Context Length | 2K | 4K | 8K |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/deepnova-ai.git

# Create development branch
git checkout -b develop

# Make changes and test
python -m pytest tests/

# Submit PR
```

---

## License

This project is licensed under the **Apache License 2.0** - see the LICENSE file for details.

Permissions:
- Commercial use
- Modification
- Distribution
- Private use

Conditions:
- License and copyright notice
- State changes

---

## Support & Contact

- Email: support@deepnova-ai.com
- Issues: GitHub Issues
- Discussions: GitHub Discussions
- Documentation: [Wiki](https://github.com/yourusername/deepnova-ai/wiki)

---

## Acknowledgments

- PyTorch team for excellent deep learning framework
- Hugging Face transformers library
- Open-source AI community

---

## Roadmap

- [ ] Mobile app (iOS/Android)
- [ ] Model quantization support
- [ ] Multi-language support
- [ ] Advanced RAG integration
- [ ] Streaming inference
- [ ] Distributed inference
- [ ] Fine-tuning tools

---

## Give us a Star!

If you find this project helpful, please consider starring it on GitHub!

# Quantization
pip install bitsandbytes>=0.42.0

# Distributed training
pip install deepspeed>=0.12.0

# API server
pip install fastapi>=0.104.0 uvicorn>=0.24.0

# Monitoring
pip install wandb>=0.16.0 tensorboard>=2.14.0

# Utilities
pip install einops>=0.7.0
```

### Step 5: Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

---

## Configuration Setup

### Create Configuration File

Create `config.yaml` in the project root:

```yaml
# DeepNova AI Configuration
model:
  name: "deepnova-moe-glm"
  version: "5.0.0"
  
  # Architecture
  dim: 4096
  n_layers: 32
  n_dense_layers: 2
  vocab_size: 102400
  
  # MoE Configuration
  n_routed_experts: 64
  n_activated_experts: 6
  n_shared_experts: 2
  moe_inter_dim: 2048
  
  # Dense Configuration
  inter_dim: 14336
  
  # Attention
  n_heads: 32
  q_lora_rank: 1536
  kv_lora_rank: 512
  qk_nope_head_dim: 128
  qk_rope_head_dim: 64
  v_head_dim: 128
  
  # Enhanced Features
  use_parallel_moe_dense: true
  parallel_moe_dense_combine: "residual_fusion"
  use_glm: true
  glm_attention_type: "bidirectional"
  use_shared_expert: true
  use_adaptive_router: true
  use_dynamic_depth: true
  
  # Training
  learning_rate: 0.0003
  warmup_steps: 2000
  weight_decay: 0.1
  max_grad_norm: 1.0
  
  # Memory
  max_seq_len: 32768
  block_size: 16
  max_num_blocks: 2048
  
  # Precision
  dtype: "bf16"
  use_amp: true

memory:
  context_file: "./deepnova_memory.json"
  max_context_tokens: 8192
```

### Set Environment Variables

Create `.env` file:

```bash
# Device Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_DTYPE=bf16

# Memory
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Checkpointing
CHECKPOINT_DIR=./checkpoints
AUTO_RESUME=true

# API
API_HOST=0.0.0.0
API_PORT=8000
```

---

## How to Run - Step by Step

### 1. Interactive Chat Mode

```bash
# Basic chat
python deepnova.py chat

# With all enhanced features
python deepnova.py chat --enhanced

# With specific model size
python deepnova.py chat --model-size enhanced

# With custom memory file
python deepnova.py chat --memory-file my_memory.json
```

**Expected Output:**
```
======================================================================
DEEPNOVA AI v5.0.0 - Interactive Chat Mode
======================================================================
Commands:
  /learn <text>     - Learn new information
  /learnfile <path> - Learn from file
  /learndir <path>  - Learn from directory
  /recall <query>   - Recall learned knowledge
  /stats            - Show statistics
  /clear            - Clear conversation context
  /list             - List learned texts
  /export <file>    - Export knowledge to file
  /quit             - Exit
======================================================================

DeepNova: Hello! I am DeepNova v5.0.0, an intelligent AI assistant 
with MoE+Dense+GLM architecture. How can I help you today?

You: 
```

### 2. Learning Mode

```bash
# Learn from text
python deepnova.py learn --text "Paris is the capital of France"

# Learn from file
echo "Important knowledge content" > knowledge.txt
python deepnova.py learn --file knowledge.txt

# Learn from directory
python deepnova.py learn --directory ./knowledge_base

# Learn with enhanced model
python deepnova.py learn --text "Complex information" --enhanced
```

### 3. Recall Mode

```bash
# Query learned knowledge
python deepnova.py recall --query "capital of France"

# With custom result limit
python deepnova.py recall --query "machine learning" --top-k 10
```

### 4. Generation Mode

```bash
# Basic generation
python deepnova.py generate --prompt "Once upon a time"

# With parameters
python deepnova.py generate \
    --prompt "The future of AI" \
    --max-tokens 200 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 50
```

### 5. API Server Mode

```bash
# Start server
python deepnova.py serve --host 0.0.0.0 --port 8000

# With enhanced features
python deepnova.py serve --enhanced --port 8000

# With specific model
python deepnova.py serve --model-path ./checkpoints/best_model
```

**Test API:**
```bash
# Chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?"}'

# Learn endpoint
curl -X POST http://localhost:8000/learn \
  -H "Content-Type: application/json" \
  -d '{"text": "Important information to remember"}'

# Stats endpoint
curl http://localhost:8000/stats
```

### 6. Training Mode

```bash
# Prepare data directory
mkdir -p ./training_data
echo "Sample training text" > ./training_data/sample.txt

# Run training
python deepnova.py train \
    --data ./training_data \
    --epochs 10 \
    --batch-size 8 \
    --lr 3e-4

# Distributed training
python deepnova.py train \
    --data ./training_data \
    --parallel \
    --enhanced \
    --checkpoint-dir ./checkpoints
```

### 7. Benchmark Mode

```bash
# Run benchmarks
python deepnova.py benchmark --iterations 10

# With specific prompt
python deepnova.py benchmark \
    --prompt "The quick brown fox jumps over the lazy dog" \
    --max-tokens 100 \
    --iterations 5
```

### 8. Testing Mode

```bash
# Run all tests
python deepnova.py test

# Expected output:
# ======================================================================
# DEEPNOVA ENHANCED UNIT TESTS
# ======================================================================
# [PASS] Model validation test
# [PASS] Enhanced model forward pass
# [PASS] Adaptive Router test
# [PASS] PagedKVCache test
# ----------------------------------------------------------------------
# Tests passed: 4
# Tests failed: 0
# ======================================================================
```

---

## Architecture Diagrams

### Diagram 1: Overall System Architecture

```
+===========================================================================+
|                         DEEPNOVA AI ARCHITECTURE                          |
|                    MoE + Dense + GLM - Version 5.0                        |
+===========================================================================+

                              INPUT TEXT
                                  │
                                  ▼
                    +-------------------------+
                    |     TOKENIZER LAYER      |
                    |  SentencePiece / BPE     |
                    |  Vocab Size: 102,400     |
                    +-------------------------+
                                  │
                                  ▼
                    +-------------------------+
                    |     EMBEDDING LAYER      |
                    |  Dim: 4096 → 7168        |
                    |  + RoPE Positional       |
                    +-------------------------+
                                  │
                                  ▼
    +=================================================================+
    |                    TRANSFORMER BLOCKS (32-61)                   |
    |                                                                 |
    |  +-----------------------------------------------------------+  |
    |  |              LAYER 0-1: DENSE LAYERS                      |  |
    |  |  +-----------+    +-----------+    +-----------+         |  |
    |  |  |   RMSNorm | -> |   MLA     | -> |   Add     |         |  |
    |  |  |   (Pre)   |    | Attention |    | Residual  |         |  |
    |  |  +-----------+    +-----------+    +-----------+         |  |
    |  |         │                                │                |  |
    |  |         ▼                                ▼                |  |
    |  |  +-----------+    +-----------+    +-----------+         |  |
    |  |  |   RMSNorm | -> |   Dense   | -> |   Add     |         |  |
    |  |  |   (Pre)   |    |    MLP    |    | Residual  |         |  |
    |  |  +-----------+    +-----------+    +-----------+         |  |
    |  +-----------------------------------------------------------+  |
    |                              │                                   |
    |                              ▼                                   |
    |  +-----------------------------------------------------------+  |
    |  |         LAYERS 2-30: PARALLEL MoE + DENSE LAYERS          |  |
    |  |                                                           |  |
    |  |                    INPUT (x)                              |  |
    |  |                       │                                   |  |
    |  |         +-------------+-------------+                     |  |
    |  |         │                           │                     |  |
    |  |         ▼                           ▼                     |  |
    |  |  +----------------+         +----------------+            |  |
    |  |  |    MoE PATH    |         |   DENSE PATH   |            |  |
    |  |  |                |         |                |            |  |
    |  |  |  Adaptive      |         |  Gate Linear   |            |  |
    |  |  |  Router        |         |  (Dim→4×Dim)   |            |  |
    |  |  |  (Top-K=6-8)   |         +-------+--------+            |  |
    |  |  +-------+--------+                 │                     |  |
    |  |          │                          │                     |  |
    |  |  +-------v--------+         +-------v--------+            |  |
    |  |  |  64-256 Experts |         |    SiLU        |            |  |
    |  |  |  (SwiGLU)      |         |  Activation    |            |  |
    |  |  +-------+--------+         +-------+--------+            |  |
    |  |          │                          │                     |  |
    |  |  +-------v--------+         +-------v--------+            |  |
    |  |  |   Weighted     |         |   Up Projection|            |  |
    |  |  |     Sum        |         |   (4×Dim→Dim)  |            |  |
    |  |  +-------+--------+         +-------+--------+            |  |
    |  |          │                          │                     |  |
    |  |          +------------+-------------+                     |  |
    |  │                       │                                   |  |
    |  |              +--------v--------+                          |  |
    |  |              |  RESIDUAL FUSION|                          |  |
    |  |              |  (Learnable     |                          |  |
    |  |              |   Weights)      |                          |  |
    |  |              +--------+--------+                          |  |
    |  |                       │                                   |  |
    |  |              +--------v--------+                          |  |
    |  |              |  Shared Expert  |                          |  |
    |  |              |   (Optional)    |                          |  |
    |  |              +--------+--------+                          |  |
    |  |                       │                                   |  |
    |  |              +--------v--------+                          |  |
    |  |              |   Add + Norm    |                          |  |
    |  |              |   (Residual)    |                          |  |
    |  |              +-----------------+                          |  |
    |  +-----------------------------------------------------------+  |
    |                              │                                   |
    |                              ▼                                   |
    |  +-----------------------------------------------------------+  |
    |  |              LAYERS 31-32: GLM LAYERS                     |  |
    |  |                                                           |  |
    |  |  GLM Attention Modes:                                     |  |
    |  |  - Bidirectional (Full attention all tokens)             |  |
    |  |  - Prefix (Bidirectional prefix + causal)                |  |
    |  |  - Sentinel (Special sentinel tokens)                    |  |
    |  +-----------------------------------------------------------+  |
    +=================================================================+
                                  │
                                  ▼
                    +-------------------------+
                    |     FINAL NORMALIZATION  |
                    |     RMSNorm + DeepNorm   |
                    +-------------------------+
                                  │
                                  ▼
                    +-------------------------+
                    |        LM HEAD          |
                    |  Linear(Dim → Vocab)     |
                    +-------------------------+
                                  │
                                  ▼
                              OUTPUT TEXT
```

### Diagram 2: GLM Attention Mechanisms

```
+===========================================================================+
|                    GLM ATTENTION MECHANISMS                               |
|              General Language Model Integration                          |
+===========================================================================+

MODE 1: BIDIRECTIONAL ATTENTION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│    Token Positions:    T0    T1    T2    T3    T4    T5                 │
│                                                                         │
│    Attention Matrix:   ┌────┬────┬────┬────┬────┬────┐                 │
│                    T0   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                    T1   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                    T2   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                    T3   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                    T4   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                    T5   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                        └────┴────┴────┴────┴────┴────┘                 │
│                                                                         │
│    Every token attends to every other token (both directions)          │
│    Best for: BERT-style tasks, text classification                      │
└─────────────────────────────────────────────────────────────────────────┘

MODE 2: PREFIX ATTENTION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│    Token Positions:    T0    T1    T2    T3    T4    T5                 │
│                   (Prefix)      (Generation)                            │
│                                                                         │
│    Attention Matrix:   ┌────┬────┬────┬────┬────┬────┐                 │
│                    T0   │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │                 │
│                    T1   │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │                 │
│                    T2   │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │                 │
│                    T3   │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │                 │
│                    T4   │ 1  │ 1  │ 1  │ 1  │ 1  │ 0  │                 │
│                    T5   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                        └────┴────┴────┴────┴────┴────┘                 │
│                                                                         │
│    Prefix tokens: bidirectional attention                               │
│    Generation tokens: causal attention (only see past)                  │
│    Best for: Conditional generation, text completion                    │
└─────────────────────────────────────────────────────────────────────────┘

MODE 3: SENTINEL ATTENTION
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│    Token Positions:    T0    T1    T2    S0    S1    T3                 │
│                   (Context)    (Sentinels)    (Target)                  │
│                                                                         │
│    Attention Matrix:   ┌────┬────┬────┬────┬────┬────┐                 │
│                    T0   │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │                 │
│                    T1   │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │                 │
│                    T2   │ 1  │ 1  │ 1  │ 0  │ 0  │ 0  │                 │
│                    S0   │ 1  │ 1  │ 1  │ 1  │ 0  │ 0  │                 │
│                    S1   │ 1  │ 1  │ 1  │ 0  │ 1  │ 0  │                 │
│                    T3   │ 1  │ 1  │ 1  │ 1  │ 1  │ 1  │                 │
│                        └────┴────┴────┴────┴────┴────┘                 │
│                                                                         │
│    Sentinels are special tokens that can attend to context              │
│    Target tokens attend to all previous tokens                          │
│    Best for: Blank filling, text infilling                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Diagram 3: Parallel MoE + Dense Fusion

```
+===========================================================================+
|                    PARALLEL MoE + DENSE ARCHITECTURE                     |
+===========================================================================+

                           INPUT (x)
                              │
              +---------------+-----------------+
              │                                 │
              v                                 v
   +----------------------+          +----------------------+
   |      MoE PATH        |          |     DENSE PATH       |
   |                      |          |                      |
   |  +----------------+  |          |  +----------------+  |
   |  | Router         |  |          |  | Gate Linear    |  |
   |  +-------+--------+  |          |  +-------+--------+  |
   |          |           |          |          |           |
   |  +-------v--------+  |          |  +-------v--------+  |
   |  | Top-K Experts  |  |          |  | SiLU Activation |  |
   |  | (k=6 or 8)     |  |          |  +-------+--------+  |
   |  +-------+--------+  |          |          |           |
   |          |           |          |  +-------v--------+  |
   |  +-------v--------+  |          |  | Up Linear      |  |
   |  | Expert Weighted|  |          |  +-------+--------+  |
   |  | Sum            |  |          |          |           |
   |  +-------+--------+  |          |  +-------v--------+  |
   |          |           |          |  | Down Linear    |  |
   |          |           |          |  +-------+--------+  |
   +----------+-----------+          +----------+-----------+
              │                                 │
              │          +----------------------+
              │          │
              v          v
   +-----------------------------+
   |      RESIDUAL FUSION        |
   |                             |
   |  Options:                   |
   |  - Add: out = a*MoE + b*Dense
   |  - Gated: out = gate*MoE + (1-gate)*Dense
   |  - Concat: out = Linear([MoE, Dense])
   |  - Residual Fusion: out = x + w1*MoE + w2*Dense
   |                             |
   +-------------+---------------+
                 │
                 v
        +----------------+
        | Shared Expert  |
        | (Optional)     |
        +-------+--------+
                │
                v
           Output (y)
```

---

## Usage Examples

### Example 1: Interactive Chat

```bash
$ python deepnova.py chat --enhanced

======================================================================
DEEPNOVA AI v5.0.0 - Interactive Chat Mode
======================================================================

DeepNova: Hello! I am DeepNova v5.0.0. How can I help you?

You: Hi! My name is John and I work as a data scientist.

DeepNova: Nice to meet you, John! As a data scientist, you must work with 
various ML models. I specialize in transformer architectures and can help 
with model training, fine-tuning, and deployment. What specific task are 
you working on?

You: /learn I specialize in PyTorch and Hugging Face transformers

DeepNova: Learned: I specialize in PyTorch and Hugging Face transformers

You: /recall What does John specialize in?

DeepNova: Found relevant information:
1. John specializes in PyTorch and Hugging Face transformers
2. John works as a data scientist

You: /stats

======================================================================
DEEPNOVA AI STATISTICS
======================================================================
  Name: DeepNova
  Version: 5.0.0
  Total Messages: 6
  Tokens Generated: 234
  Active Features: MoE, Dense, GLM, Adaptive Router
----------------------------------------------------------------------
  Memory:
    Short-term: 6 messages
    Important Facts: 2
    Entities Tracked: 2
----------------------------------------------------------------------
  Learning:
    Total Learned: 1
    Knowledge Graph Nodes: 8
======================================================================
```

### Example 2: API Server Usage

```python
# client.py
import requests

# Chat
response = requests.post(
    "http://localhost:8000/chat",
    json={"prompt": "Explain quantum computing"}
)
print(response.json()["response"])

# Learn
response = requests.post(
    "http://localhost:8000/learn",
    json={"text": "Quantum computers use qubits instead of bits"}
)
print(response.json())

# Get stats
response = requests.get("http://localhost:8000/stats")
print(response.json())
```

---

## Model Comparison Matrix

| Feature | Standard MoE | DeepNova Lite | DeepNova Enhanced |
|---------|-------------|---------------|-------------------|
| Total Parameters | 7B | 15B | 25B |
| Active Parameters | 7B | 2.5B | 3.2B |
| Sparsity | 0% | 83% | 87% |
| Expert Count | 8 | 64 | 256 |
| Activated Experts | 2 | 6 | 8 |
| Hidden Dimension | 4096 | 2048 | 7168 |
| Layers | 32 | 27 | 61 |
| Attention Heads | 32 | 16 | 128 |
| Max Sequence Length | 8K | 32K | 131K |
| Parallel MoE+Dense | No | Yes | Yes |
| GLM Integration | No | No | Yes |
| Adaptive Router | No | Yes | Yes |
| Dynamic Depth | No | No | Yes |
| Multi-Token Prediction | No | No | Optional |

---

## Feature Documentation

### 1. Parallel MoE + Dense

Executes MoE and Dense paths simultaneously, then fuses results.

**Combine Modes:**
- `add`: Weighted sum (fastest)
- `gated`: Learnable gate between paths
- `concat`: Concatenate then project
- `residual_fusion`: x + w1*MoE + w2*Dense (recommended)

### 2. GLM (General Language Model)

Three attention modes:
- `bidirectional`: Full attention (BERT-style)
- `prefix`: Bidirectional prefix + causal generation
- `sentinel`: Special tokens for blank filling

### 3. Adaptive Router

Features:
- Learnable temperature (0.5 to 2.0)
- Dynamic expert bias
- Jitter for exploration
- Load balancing loss

### 4. Dynamic Depth

Intelligent layer skipping based on learned confidence scores. Benefits:
- Up to 30% faster inference
- Adaptive computation
- No quality degradation

---

## Troubleshooting Guide

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, enable gradient checkpointing, use FP16 |
| Flash Attention not found | `pip install flash-attn --no-build-isolation` |
| Import errors | Check Python version (3.9+ required) |
| Slow inference | Enable FP16, use smaller model, enable KV cache |
| Tokenizer errors | Download sentencepiece model or use HF fallback |

### Debug Commands

```bash
# Check GPU memory
nvidia-smi

# Monitor memory usage
python -c "import torch; print(torch.cuda.memory_summary())"

# Run with debug logging
python deepnova.py chat --debug

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Performance Tuning

### Optimization Flags

```python
# In ModelArgs
args.use_flash_attn = True      # Faster attention
args.use_fused_linear = True    # Fused linear layers
args.use_triton_kernels = True  # Triton optimizations
args.gradient_checkpointing = True  # Memory efficient
args.use_amp = True             # Mixed precision
```

### Batch Size Recommendations

| GPU | Batch Size (Seq Len 4096) |
|-----|--------------------------|
| 8GB | 1-2 |
| 16GB | 4-8 |
| 24GB | 8-16 |
| 40GB | 16-32 |
| 80GB | 32-64 |

---

## API Reference

### ModelArgs Class

```python
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    use_parallel_moe_dense: bool = True
    use_glm: bool = True
    use_adaptive_router: bool = True
    use_dynamic_depth: bool = True
```

### DeepNovaAI Class

```python
class DeepNovaAI:
    def chat(self, user_input: str) -> str
    def learn(self, text: str) -> Dict
    def recall(self, query: str) -> List[Dict]
    def get_stats(self) -> Dict
```

---

## License

```
Copyright 2026DeepNova AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

<div align="center">
  <p><strong>DeepNova AI</strong> | Built with PyTorch | Apache 2.0 License</p>
  <p><a href="https://github.com/deepnova-ai/deepnova">GitHub</a></p>
</div>"# deepnova" 
