# DeepNova AI

Một trợ lý AI chuyên nghiệp với kiến trúc MoE (Mixture of Experts) + Dense + GLM Transformer, cung cấp nhiều giao diện: desktop, web và dòng lệnh.

---

## Danh mục

- [Tính năng](#tinh-nang)
- [Cấu trúc dự án](#cau-truc-du-an)
- [Yêu cầu hệ thống](#yeu-cau-he-thong)
- [Cài đặt](#cai-dat)
- [Khởi động nhanh](#khoi-dong-nhanh)
- [Hướng dẫn sử dụng](#huong-dan-su-dung)
- [Kiến trúc](#kien-truc)
- [Cấu hình](#cau-hinh)
- [Tối ưu hiệu suất](#toi-uu-hieu-suat)
- [Xử lý sự cố](#xu-ly-su-co)
- [Giấy phép](#giay-phep)

---

## Tính năng

### Giao diện

| Giao diện | Mô tả |
|-----------|-------|
| **Desktop GUI** | Giao diện Tkinter nền trắng, nút xanh |
| **Web** | Máy chủ Flask với REST API |
| **Chat đơn giản** | Giao diện trò chuyện tối giản |
| **DevTool** | Công cụ phát triển tích hợp terminal |
| **CLI** | Giao diện dòng lệnh |

### Kiến trúc

- **MoE** - Hỗn hợp chuyên gia
- **Dense Layers** - Lớp đặc
- **GLM Transformer** - Mô hình nền tảng
- **Adaptive Router** - Định tuyến thích ứng
- **Dynamic Depth** - Bỏ qua lớp thông minh

### Hỗ trợ phần cứng

- NVIDIA CUDA
- Apple Metal (MPS)
- Intel XPU
- CPU fallback

---

## Cấu trúc dự án

```
DeepNova-AI/
│
├── src/
│   ├── app.py              # Ứng dụng desktop chính
│   ├── cmd.py              # Công cụ phát triển dòng lệnh
│   ├── code.py             # Công cụ phát triển chuyên nghiệp
│   ├── gui_chat.py         # Giao diện chat đơn giản
│   ├── gui_setting.py      # Giao diện style VSCode
│   ├── model.py            # Mô hình AI lõi (MoE + Dense + GLM)
│   ├── training.py         # Giao diện huấn luyện PyQt6
│   ├── web.py              # Máy chủ web Flask
│   │
│   ├── logs/               # Thư mục log
│   └── checkpoints/        # Checkpoints mô hình
│
├── requirements.txt        # Thư viện phụ thuộc
└── README.md               # Tài liệu này
```

### Mô tả file

| File | Công dụng |
|------|-----------|
| `app.py` | Ứng dụng desktop chính - giao diện trắng xanh |
| `cmd.py` | Công cụ dòng lệnh với terminal tích hợp |
| `code.py` | IDE nhẹ với editor code, console Python |
| `gui_chat.py` | Giao diện chat đơn giản, nhẹ |
| `gui_setting.py` | Giao diện style VSCode đầy đủ tính năng |
| `model.py` | Model AI lõi - chứa Transformer, MoE, tokenizer |
| `training.py` | Giao diện huấn luyện với PyQt6 |
| `web.py` | Web server Flask, REST API |

---

## Yêu cầu hệ thống

### Tối thiểu

| Thành phần | Yêu cầu |
|------------|---------|
| **HĐH** | Linux, macOS 12+, Windows 10/11 |
| **Python** | 3.9+ |
| **CPU** | 4 nhân |
| **RAM** | 8 GB |
| **Storage** | 20 GB |

### Khuyến nghị

| Thành phần | Yêu cầu |
|------------|---------|
| **Python** | 3.10+ |
| **RAM** | 32 GB+ |
| **GPU** | NVIDIA 24GB VRAM |

---

## Cài đặt

### Bước 1: Clone repository

```bash
git clone https://github.com/yourusername/deepnova-ai.git
cd deepnova-ai
```

### Bước 2: Tạo môi trường ảo

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện

```bash
pip install --upgrade pip

# PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Hoặc PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Thư viện cốt lõi
pip install transformers>=4.36.0 sentencepiece>=0.1.99
pip install numpy>=1.24.0 tqdm>=4.66.0 psutil>=5.9.0
pip install flask flask-cors

# PyQt6 cho training.py (tùy chọn)
pip install PyQt6
```

### Bước 4: Kiểm tra

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## Khởi động nhanh

```bash
# Giao diện desktop chính
python src/app.py

# Giao diện chat đơn giản
python src/gui_chat.py

# Giao diện style VSCode
python src/gui_setting.py

# Máy chủ web
python src/web.py
# Mở trình duyệt: http://localhost:5000

# Công cụ phát triển
python src/cmd.py

# Giao diện huấn luyện
python src/training.py
```

---

## Hướng dẫn sử dụng

### Giao diện Desktop (app.py)

1. Chạy `python src/app.py`
2. Chờ mô hình tải
3. Nhập tin nhắn, nhấn **Send** hoặc `Ctrl+Enter`
4. Điều chỉnh tham số ở thanh bên

### Web Interface (web.py)

```bash
# Khởi động server
python src/web.py

# API endpoints
curl http://localhost:5000/api/status           # Trạng thái
curl -X POST http://localhost:5000/api/load     # Tải model
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Xin chào"}'
curl -X POST http://localhost:5000/api/clear    # Xóa bộ nhớ
```

### Lệnh Chat

| Lệnh | Mô tả |
|------|-------|
| `/clear` | Xóa đoạn chat |
| `/stats` | Xem thống kê |
| `/learn <text>` | Học thông tin mới |
| `/recall <query>` | Tra cứu kiến thức |
| `/help` | Hiển thị trợ giúp |

---

## Kiến trúc

### Sơ đồ tổng thể

```
INPUT TEXT
    │
    ▼
┌─────────────────┐
│   TOKENIZER     │
│ SentencePiece   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   EMBEDDING     │
│ + RoPE          │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│          TRANSFORMER BLOCKS             │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │   LAYER 0-1: DENSE MLP          │   │
│  │   Attention → RMSNorm → Dense   │   │
│  └─────────────────────────────────┘   │
│                  │                      │
│  ┌─────────────────────────────────┐   │
│  │   LAYERS 2-30: MoE + DENSE      │   │
│  │                                  │   │
│  │    INPUT ──┬──────┬─────────┐   │   │
│  │            │      │         │   │   │
│  │            ▼      ▼         ▼   │   │
│  │          MoE   Dense    Shared  │   │
│  │         Path   Path     Expert  │   │
│  │            │      │         │   │   │
│  │            └──────┴─────────┘   │   │
│  │                  │              │   │
│  │            RESIDUAL FUSION      │   │
│  └─────────────────────────────────┘   │
│                  │                      │
│  ┌─────────────────────────────────┐   │
│  │   LAYERS 31-32: GLM             │   │
│  │   Bidirectional / Prefix        │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   RMSNorm       │
│   + LM HEAD     │
└────────┬────────┘
         │
         ▼
   OUTPUT TEXT
```

### Các chế độ GLM

| Chế độ | Mô tả | Ứng dụng |
|--------|-------|----------|
| **Bidirectional** | Attention hai chiều | Phân loại văn bản |
| **Prefix** | Prefix 2 chiều + sinh nhân quả | Sinh có điều kiện |
| **Sentinel** | Token đặc biệt | Điền vào chỗ trống |

---

## Cấu hình

### File cấu hình mẫu (config.yaml)

```yaml
model:
  name: "deepnova-moe"
  version: "5.0.0"
  
  # Kiến trúc
  dim: 4096
  n_layers: 32
  n_dense_layers: 2
  
  # MoE
  n_routed_experts: 64
  n_activated_experts: 6
  
  # Attention
  n_heads: 32
  max_seq_len: 32768
  
  # Tính năng nâng cao
  use_parallel_moe_dense: true
  use_glm: true
  use_adaptive_router: true
  use_dynamic_depth: true
  
  # Huấn luyện
  learning_rate: 0.0003
  warmup_steps: 2000
  weight_decay: 0.1
```

### Biến môi trường (.env)

```bash
CUDA_VISIBLE_DEVICES=0
TORCH_DTYPE=bf16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
CHECKPOINT_DIR=./checkpoints
```

---

## Tối ưu hiệu suất

### GPU

```python
# Bật mixed precision
torch.set_float32_matmul_precision('high')

# Dùng Flash Attention
args.use_flash_attn = True
args.use_fused_linear = True
```

### Bộ nhớ

```python
# Gradient checkpointing
args.gradient_checkpointing = True

# Batch size nhỏ
batch_size = 1

# Dọn bộ nhớ
torch.cuda.empty_cache()
```

### Kích thước batch theo GPU

| GPU | Batch size (seq_len=4096) |
|-----|--------------------------|
| 8GB | 1-2 |
| 16GB | 4-8 |
| 24GB | 8-16 |
| 40GB | 16-32 |

---

## Tài liệu API

### ModelArgs

```python
@dataclass
class ModelArgs:
    dim: int = 4096                 # Chiều ẩn
    n_layers: int = 32              # Số lớp
    n_heads: int = 32               # Số đầu attention
    n_routed_experts: int = 64      # Số chuyên gia
    n_activated_experts: int = 6    # Số chuyên gia kích hoạt
    max_seq_len: int = 32768        # Độ dài chuỗi tối đa
    
    # Tính năng nâng cao
    use_parallel_moe_dense: bool = True
    use_glm: bool = True
    use_adaptive_router: bool = True
    use_dynamic_depth: bool = True
```

### DeepNovaAI

```python
class DeepNovaAI:
    def chat(self, user_input: str, **kwargs) -> str
    def learn(self, text: str, source: str = "user") -> Dict
    def recall(self, query: str, top_k: int = 5) -> List[Dict]
    def learn_from_file(self, file_path: str) -> List[Dict]
    def clear_context(self, keep_important: bool = True) -> None
    def get_stats(self) -> Dict
    def get_model_info(self) -> Dict
```

---

## Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|-----------|
| **CUDA out of memory** | Giảm batch size, bật gradient checkpointing |
| **Model không tải** | Kiểm tra logs: `cat logs/deepnova.log` |
| **Import error** | Kiểm tra Python version (cần 3.9+) |
| **Web không chạy** | `pip install flask flask-cors` |
| **GUI không mở** | `sudo apt-get install python3-tk` (Linux) |

### Lệnh gỡ lỗi

```bash
# Kiểm tra GPU
nvidia-smi

# Dọn bộ nhớ
python -c "import torch; torch.cuda.empty_cache()"

# Xem logs
tail -f logs/deepnova.log
```

---

## Giấy phép

```
Apache License 2.0

Copyright 2026 DeepNova Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
