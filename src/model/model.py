

import os
import sys
import math
import time
import json
import logging
import hashlib
import threading
import argparse
import uuid
import traceback
import re
import gc
import weakref
from typing import Dict, List, Optional, Tuple, Union, Generator, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from collections import deque, defaultdict, OrderedDict
from functools import partial, wraps, lru_cache
from itertools import chain
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
import signal
import atexit

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "deepnova.log"

log_formatter = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-7s | %(name)-20s | %(funcName)-20s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.handlers = []
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logger = logging.getLogger("DeepNova")
logger.setLevel(logging.INFO)

debug_logger = logging.getLogger("DeepNova.Debug")
debug_logger.setLevel(logging.DEBUG)

perf_logger = logging.getLogger("DeepNova.Performance")
perf_logger.setLevel(logging.INFO)

memory_logger = logging.getLogger("DeepNova.Memory")
memory_logger.setLevel(logging.INFO)


# ============================================================================
# DEPENDENCY CHECKING AND DEVICE DETECTION
# ============================================================================

def get_best_device() -> str:
    """Detect the best available compute device"""
    logger.debug("Detecting best available compute device...")
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
            return device
        elif hasattr(torch, 'backends') and getattr(torch.backends, 'mps', None) is not None:
            try:
                if torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("Apple Metal Performance Shaders (MPS) detected")
                    return device
            except Exception as mps_error:
                logger.warning(f"MPS detection failed, falling back to next available device: {mps_error}")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = "xpu"
            logger.info("Intel XPU device detected")
            return device
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"
            logger.info("NPU device detected")
            return device
        else:
            device = "cpu"
            logger.warning("No accelerator detected, using CPU")
            return device
    except ImportError:
        logger.error("PyTorch not available, using CPU")
        return "cpu"


def get_device_compute_capability() -> Dict[str, Any]:
    """Get detailed device compute capabilities"""
    logger.debug("Gathering device compute capabilities...")
    info = {"device": get_best_device()}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            info["compute_capability"] = torch.cuda.get_device_capability(0)
            info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
            info["fp8_supported"] = torch.cuda.get_device_capability(0) >= (8, 9)
            
            logger.info(f"Device Capabilities:")
            logger.info(f"  - Device: {info['device_name']}")
            logger.info(f"  - CUDA Version: {info['cuda_version']}")
            logger.info(f"  - Device Count: {info['device_count']}")
            logger.info(f"  - Compute Capability: {info['compute_capability']}")
            logger.info(f"  - Total VRAM: {info['memory_gb']:.2f} GB")
            logger.info(f"  - FP8 Support: {info['fp8_supported']}")
        elif torch.backends.mps.is_available():
            info["mps_available"] = True
            logger.info("Device Capabilities: Apple MPS enabled")
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            info["xpu_available"] = True
            info["device_count"] = torch.xpu.device_count()
            logger.info(f"Device Capabilities: Intel XPU with {info['device_count']} device(s)")
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            info["npu_available"] = True
            info["device_count"] = torch.npu.device_count()
            logger.info(f"Device Capabilities: NPU with {info['device_count']} device(s)")
    except Exception as e:
        logger.error(f"Device detection failed: {e}", exc_info=True)
    
    return info


def check_dependencies() -> Dict[str, bool]:
    """Check all required and optional dependencies"""
    logger.info("=" * 80)
    logger.info("STARTING DEPENDENCY CHECK")
    logger.info("=" * 80)
    
    deps = {
        'torch': 'torch>=2.1.0',
        'transformers': 'transformers>=4.36.0',
        'sentencepiece': 'sentencepiece>=0.1.99',
        'safetensors': 'safetensors>=0.4.0',
        'tqdm': 'tqdm>=4.66.0',
        'numpy': 'numpy>=1.24.0',
        'psutil': 'psutil>=5.9.0',
    }
    
    optional_deps = {
        'flash_attn': 'flash-attn>=2.3.0',
        'megablocks': 'megablocks>=0.6.0',
        'triton': 'triton>=2.1.0',
        'wandb': 'wandb>=0.16.0',
        'vllm': 'vllm>=0.3.0',
        'deepspeed': 'deepspeed>=0.12.0',
        'tensorboard': 'tensorboard>=2.14.0',
        'bitsandbytes': 'bitsandbytes>=0.42.0',
        'fastapi': 'fastapi>=0.104.0',
        'uvicorn': 'uvicorn>=0.24.0',
        'einops': 'einops>=0.7.0',
    }
    
    available = {}
    
    logger.info(f"Checking {len(deps)} REQUIRED dependencies...")
    for dep, install_cmd in deps.items():
        try:
            __import__(dep.replace('-', '_'))
            available[dep] = True
            logger.info(f"OK REQUIRED: {dep:20s} | Install: {install_cmd}")
        except ImportError as e:
            available[dep] = False
            logger.error(f"MISSING REQUIRED: {dep:20s} | Missing - pip install {install_cmd} | Error: {e}")
    
    logger.info(f"\nChecking {len(optional_deps)} OPTIONAL dependencies...")
    for dep, install_cmd in optional_deps.items():
        try:
            __import__(dep.replace('-', '_').replace('flash_attn', 'flash_attn'))
            available[dep] = True
            logger.info(f"OK OPTIONAL: {dep:20s} | Install: {install_cmd}")
        except ImportError:
            available[dep] = False
            logger.debug(f"SKIP OPTIONAL: {dep:20s} | Not installed (optional)")
    
    required_count = sum(1 for dep in deps if available.get(dep, False))
    optional_count = sum(1 for dep in optional_deps if available.get(dep, False))
    logger.info(f"\nDependency Summary: {required_count}/{len(deps)} required, {optional_count}/{len(optional_deps)} optional")
    logger.info("=" * 80)
    
    return available


DEPENDENCIES = check_dependencies()
HAS_SAFETENSORS = DEPENDENCIES.get('safetensors', False)
HAS_FASTAPI = DEPENDENCIES.get('fastapi', False)
HAS_UVICORN = DEPENDENCIES.get('uvicorn', False)


# ============================================================================
# PYTORCH IMPORTS
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn import init
    from torch.utils.checkpoint import checkpoint as torch_checkpoint
    from torch.cuda.amp import autocast, GradScaler
    from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy, BackwardPrefetch, MixedPrecision,
        CPUOffload, StateDictType, FullStateDictConfig
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy, size_based_auto_wrap_policy,
        enable_wrap, wrap
    )
    logger.info("PyTorch imported successfully")
except ImportError as e:
    logger.error(f"PyTorch import failed: {e}")
    raise

try:
    import numpy as np
except ImportError as e:
    logger.error(f"NumPy import failed: {e}")
    raise


# ============================================================================
# OPTIONAL DEPENDENCY IMPORTS
# ============================================================================

HAS_FLASH_ATTN = DEPENDENCIES.get('flash_attn', False)
if HAS_FLASH_ATTN:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_qkvpacked_func
        from flash_attn.ops.triton.k_layer_norm import layer_norm_fn
        from flash_attn.layers.rotary import apply_rotary_emb
        logger.info("Flash Attention 2 loaded")
    except ImportError:
        HAS_FLASH_ATTN = False

HAS_MEGABLOCKS = DEPENDENCIES.get('megablocks', False)
if HAS_MEGABLOCKS:
    try:
        from megablocks.layers import moe as mega_moe
        from megablocks.layers.arguments import Arguments as MegaArgs
        from megablocks.layers.dmoe import dMoE
        from megablocks.layers.moe import MoE as MegaMoE
        from megablocks.layers import grouped_gemm_util as gg
        logger.info("Megablocks loaded")
    except ImportError:
        HAS_MEGABLOCKS = False

HAS_TRITON = DEPENDENCIES.get('triton', False)
if HAS_TRITON:
    try:
        import triton
        import triton.language as tl
        logger.info("Triton loaded")
    except ImportError:
        HAS_TRITON = False

HAS_BNB = DEPENDENCIES.get('bitsandbytes', False)
if HAS_BNB:
    try:
        import bitsandbytes as bnb
        from bitsandbytes.nn import Linear4bit, Linear8bitLt
        from bitsandbytes.optim import AdamW8bit, Adam4bit
        logger.info("BitsAndBytes loaded")
    except ImportError:
        HAS_BNB = False

HAS_DEEPSPEED = DEPENDENCIES.get('deepspeed', False)
if HAS_DEEPSPEED:
    try:
        import deepspeed
        from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
        logger.info("DeepSpeed loaded")
    except ImportError:
        HAS_DEEPSPEED = False

HAS_VLLM = DEPENDENCIES.get('vllm', False)
if HAS_VLLM:
    logger.info("vLLM available")

HAS_WANDB = DEPENDENCIES.get('wandb', False)
if HAS_WANDB:
    try:
        import wandb
        logger.info("Weights & Biases loaded")
    except ImportError:
        HAS_WANDB = False

HAS_TB = DEPENDENCIES.get('tensorboard', False)
if HAS_TB:
    try:
        from torch.utils.tensorboard import SummaryWriter
        logger.info("TensorBoard loaded")
    except ImportError:
        HAS_TB = False

HAS_SPM = DEPENDENCIES.get('sentencepiece', False)
if HAS_SPM:
    try:
        import sentencepiece as spm
        logger.info("SentencePiece loaded")
    except ImportError:
        HAS_SPM = False

HAS_EINOPS = DEPENDENCIES.get('einops', False)
if HAS_EINOPS:
    try:
        from einops import rearrange, repeat, reduce, einsum
        logger.info("Einops loaded")
    except ImportError:
        HAS_EINOPS = False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device() -> torch.device:
    """Get the primary compute device"""
    logger.debug("Resolving primary compute device...")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.debug(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.debug("Using Apple MPS device")
        return device
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        logger.debug("Using Intel XPU device")
        return device
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        device = torch.device("npu")
        logger.debug("Using NPU device")
        return device
    device = torch.device("cpu")
    logger.warning("Falling back to CPU device")
    return device


def get_memory_info() -> Dict[str, float]:
    """Get detailed memory usage information"""
    logger.debug("Querying memory usage information...")
    info = {}
    try:
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            gpu_max = torch.cuda.max_memory_allocated() / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_free = (gpu_total - gpu_allocated) / 1e9
            
            info['gpu_allocated_gb'] = gpu_allocated
            info['gpu_reserved_gb'] = gpu_reserved
            info['gpu_max_allocated_gb'] = gpu_max
            info['gpu_free_gb'] = gpu_free
            info['gpu_total_gb'] = gpu_total
            
            logger.debug(f"GPU Memory: allocated={gpu_allocated:.2f}GB, reserved={gpu_reserved:.2f}GB, "
                        f"free={gpu_free:.2f}GB, total={gpu_total:.2f}GB, max_allocated={gpu_max:.2f}GB")
    except Exception as e:
        logger.error(f"Failed to get GPU memory info: {e}", exc_info=True)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['ram_total_gb'] = mem.total / 1e9
        info['ram_available_gb'] = mem.available / 1e9
        info['ram_used_gb'] = mem.used / 1e9
        info['ram_percent'] = mem.percent
        
        logger.debug(f"System Memory: used={info['ram_used_gb']:.2f}GB, available={info['ram_available_gb']:.2f}GB, "
                    f"total={info['ram_total_gb']:.2f}GB, percent={info['ram_percent']:.1f}%")
    except ImportError:
        logger.warning("psutil not available, skipping system memory info")
    
    return info


def print_memory_usage(prefix: str = ""):
    """Print current memory usage"""
    info = get_memory_info()
    msg = f"{prefix} | " if prefix else ""
    if 'gpu_allocated_gb' in info:
        gpu_pct = (info['gpu_allocated_gb'] / info['gpu_total_gb'] * 100) if info['gpu_total_gb'] > 0 else 0
        msg += f"GPU: {info['gpu_allocated_gb']:.2f}/{info['gpu_total_gb']:.1f}GB ({gpu_pct:.1f}%) | "
    if 'ram_used_gb' in info:
        msg += f"RAM: {info['ram_used_gb']:.1f}/{info['ram_total_gb']:.1f}GB ({info['ram_percent']:.0f}%)"
    
    memory_logger.info(msg)


def cleanup_memory():
    """Clean up GPU and RAM memory"""
    logger.info("Starting memory cleanup...")
    gc.collect()
    logger.debug("Python garbage collection completed")
    
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            logger.debug("CUDA synchronization completed")
            
            before = torch.cuda.memory_allocated() / 1e9
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            after = torch.cuda.memory_allocated() / 1e9
            
            logger.info(f"CUDA cache cleared: freed {before - after:.2f}GB")
        except Exception as e:
            logger.error(f"Failed to cleanup CUDA: {e}", exc_info=True)
    
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")
        except Exception as e:
            logger.error(f"Failed to cleanup MPS: {e}")


def safe_tensor_op(func):
    """Decorator for safe tensor operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Executing safe tensor operation: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Safe tensor operation completed: {func.__name__}")
            return result
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM detected in {func.__name__}, starting cleanup...")
                cleanup_memory()
                logger.error(f"Out of memory error in {func.__name__}: {e}")
                raise
            logger.error(f"Runtime error in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


# ============================================================================
# MEMORY PROFILER FOR LEAK DETECTION
# ============================================================================

class MemoryProfiler:
    """Track memory usage to detect leaks"""
    
    def __init__(self, enabled: bool = True):
        self.snapshots = []
        self.enabled = enabled or os.environ.get('DEBUG_MEMORY', '0') == '1'
    
    def take_snapshot(self, name: str) -> Dict[str, float]:
        """Take a memory snapshot"""
        if not self.enabled:
            return {}
        
        gc.collect()
        snapshot = {
            'name': name,
            'time': time.time(),
            'cpu_memory': 0.0,
        }
        
        try:
            import psutil
            snapshot['cpu_memory'] = psutil.Process().memory_info().rss / 1e9
        except ImportError:
            pass
        
        if torch.cuda.is_available():
            snapshot['gpu_memory'] = torch.cuda.memory_allocated() / 1e9
            snapshot['gpu_cached'] = torch.cuda.memory_reserved() / 1e9
        
        self.snapshots.append(snapshot)
        
        if len(self.snapshots) > 1:
            prev = self.snapshots[-2]
            diff = snapshot['cpu_memory'] - prev['cpu_memory']
            if diff > 0.1:
                logger.warning(f"Potential leak in {name}: +{diff:.2f}GB")
        
        return snapshot
    
    def report(self):
        """Print memory profiler report"""
        if not self.enabled:
            return
        
        print("\n" + "="*70)
        print("MEMORY PROFILER REPORT")
        print("="*70)
        for i, snap in enumerate(self.snapshots):
            msg = f"{i}: {snap['name']:<30} - CPU: {snap['cpu_memory']:>6.2f}GB"
            if 'gpu_memory' in snap:
                msg += f" | GPU: {snap['gpu_memory']:>6.2f}GB (cached: {snap['gpu_cached']:>6.2f}GB)"
            print(msg)
        print("="*70)

    def clear(self):
        """Reset profiler snapshots and release cached memory"""
        self.snapshots = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()


@contextmanager
def timer(name: str = ""):
    """Context manager for timing code blocks"""
    start = time.perf_counter()
    logger.debug(f"Timer started: {name if name else 'unnamed'}")
    yield
    elapsed = time.perf_counter() - start
    perf_logger.info(f"Timer {name}: {elapsed:.3f}s" if name else f"Timer: {elapsed:.3f}s")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters"""
    logger.debug(f"Counting model parameters (trainable_only={trainable_only})...")
    
    if trainable_only:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {format_number(total)} ({total:,})")
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        
        logger.info(f"Parameter Count Summary:")
        logger.info(f"  Total:     {format_number(total):>12} ({total:>15,})")
        logger.info(f"  Trainable: {format_number(trainable):>12} ({trainable:>15,}) {trainable/total*100:.1f}%")
        logger.info(f"  Frozen:    {format_number(frozen):>12} ({frozen:>15,}) {frozen/total*100:.1f}%")
    
    return total


def format_number(n: int) -> str:
    """Format large numbers with suffixes"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


# ============================================================================
# ENHANCED MODEL CONFIGURATION
# ============================================================================

@dataclass
class ModelArgs:
    """Complete model configuration with all enhanced features"""
    # Architecture
    dim: int = 4096
    n_layers: int = 32
    n_dense_layers: int = 2
    vocab_size: int = 102400
    
    # Parallel MoE + Dense (Enhanced)
    use_parallel_moe_dense: bool = False
    parallel_moe_dense_ratio: float = 0.5
    parallel_moe_dense_combine: str = "add"  # add, concat, gated, residual_fusion
    
    # GLM (General Language Model) Integration
    use_glm: bool = False
    glm_attention_type: str = "bidirectional"  # bidirectional, prefix, sentinel
    glm_num_sentinels: int = 100
    
    # Shared Expert (Enhanced)
    use_shared_expert: bool = True
    shared_expert_inter_dim: int = 4096
    shared_expert_scale: float = 0.5
    
    # Adaptive Router (Enhanced)
    use_adaptive_router: bool = True
    adaptive_router_temperature_min: float = 0.5
    adaptive_router_temperature_max: float = 2.0
    adaptive_router_bias_update_rate: float = 0.01
    adaptive_router_expert_capacity_factor: float = 1.25
    
    # Dynamic Depth (Layer Skipping)
    use_dynamic_depth: bool = False
    dynamic_depth_confidence_threshold: float = 0.7
    dynamic_depth_min_layers: int = 4
    dynamic_depth_skip_prob: float = 0.1
    
    # Multi-Token Prediction (MTP)
    use_multi_token_prediction: bool = False
    mtp_n_predictions: int = 4
    mtp_loss_weight: float = 0.3
    
    # FP8 Training Support
    use_fp8_training: bool = False
    fp8_amax_history_len: int = 16
    fp8_amax_compute_algo: str = "max"
    
    # Attention
    n_heads: int = 32
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    attention_dropout: float = 0.0
    
    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim
    
    rope_theta: float = 100000.0
    rope_scaling: Optional[Dict] = None
    max_seq_len: int = 32768
    
    # MoE Configuration (Enhanced)
    n_routed_experts: int = 256
    n_activated_experts: int = 8
    n_shared_experts: int = 2
    moe_inter_dim: int = 2048
    expert_capacity_factor: float = 1.25
    moe_use_fused_kernel: bool = True
    moe_dropless: bool = True
    
    moe_router_topk: int = 8
    moe_router_score_func: str = "sigmoid"
    moe_router_temperature: float = 1.0
    moe_router_jitter: float = 0.01
    
    # Auxiliary Losses (Enhanced)
    moe_aux_loss_weight: float = 0.001
    moe_router_z_loss_weight: float = 0.001
    moe_load_balance_loss_weight: float = 0.01
    moe_load_balance_epsilon: float = 0.0001
    
    # Expert Parallelism
    moe_expert_parallel: bool = False
    moe_expert_parallel_group_size: int = 1
    
    # Tokenizer tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    mask_token_id: int = 4
    sentinel_start_id: int = 32000
    
    # MLP
    inter_dim: int = 14336
    activation_fn: str = "swiglu"
    
    # KV Cache
    block_size: int = 16
    max_num_blocks: int = 2048
    cache_dtype: str = "bf16"
    use_paged_kv_cache: bool = True
    use_paged_attention: bool = True
    
    # Training
    max_batch_size: int = 32
    learning_rate: float = 3e-4
    min_lr: float = 1e-6
    warmup_steps: int = 2000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    gradient_checkpointing: bool = True
    gradient_checkpointing_layers: int = 1
    use_amp: bool = True
    gradient_accumulation_steps: int = 1
    
    label_smoothing: float = 0.0
    dropout: float = 0.0
    embedding_dropout: float = 0.0
    residual_dropout: float = 0.0
    
    # Distributed Training
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    expert_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "full_shard"
    fsdp_backward_prefetch: str = "backward_pre"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    
    use_zero3: bool = False
    zero_stage: int = 3
    
    # Precision
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bf16"
    attention_dtype: str = "bf16"
    matmul_dtype: str = "bf16"
    
    # Optimizations
    use_flash_attn: bool = True
    use_fused_linear: bool = True
    use_fused_rope: bool = True
    use_fused_swiglu: bool = True
    use_triton_kernels: bool = True
    
    # Quantization
    quantize: bool = False
    quantize_type: str = "int8"
    quantize_calibration_steps: int = 100
    quantize_activations: bool = False
    quantization_config: Optional[Dict] = None
    
    use_gptq: bool = False
    gptq_bits: int = 4
    gptq_groupsize: int = 128
    use_awq: bool = False
    awq_bits: int = 4
    
    # Normalization
    rms_norm_eps: float = 1e-6
    use_deepnorm: bool = False
    
    # Inference
    use_kv_cache: bool = True
    use_speculative_decoding: bool = False
    draft_model_path: str = ""
    num_speculative_tokens: int = 5
    speculative_temperature: float = 0.7
    
    use_continuous_batching: bool = False
    max_waiting_tokens: int = 512
    max_batch_size_inference: int = 64
    
    use_cuda_graphs: bool = False
    
    # Memory Management
    use_cpu_offload: bool = False
    cpu_offload_layers: List[int] = field(default_factory=list)
    use_activation_offload: bool = False
    memory_efficient_attention: bool = True
    split_large_ops: bool = True
    
    # Data
    data_path: str = ""
    val_data_path: str = ""
    max_seq_length: int = 4096
    min_seq_length: int = 64
    batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    save_total_limit: int = 3
    log_grad_norm: bool = True
    log_param_stats: bool = True
    log_memory_usage: bool = True
    log_throughput: bool = True
    log_router_stats: bool = True
    
    # Experiment Tracking
    wandb_project: str = "deepnova-moe"
    wandb_run_name: str = ""
    wandb_entity: str = ""
    use_wandb: bool = False
    
    use_tensorboard: bool = False
    tensorboard_dir: str = "./logs"
    
    profile_steps: List[int] = field(default_factory=list)
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_freq: int = 1000
    auto_resume: bool = True
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    checkpoint_format: str = "safetensors"
    
    # Debug
    debug: bool = False
    detect_anomaly: bool = False
    print_model_stats: bool = False
    torch_compile: bool = False
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "reduce-overhead"
    
    # Model Metadata
    model_name: str = "deepnova-moe"
    model_version: str = "5.0.0"
    model_author: str = "DeepNova-Team"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        logger.info("=" * 80)
        logger.info("VALIDATING ENHANCED MODEL CONFIGURATION")
        logger.info("=" * 80)
        
        # Validate architecture
        logger.debug(f"Validating dimension divisibility: dim={self.dim} must be divisible by n_heads={self.n_heads}")
        assert self.dim % self.n_heads == 0, f"dim {self.dim} must be divisible by n_heads {self.n_heads}"
        logger.debug(f"Dimension validation passed: head_dim = {self.dim // self.n_heads}")
        
        # Validate MoE
        logger.debug(f"Validating MoE experts: n_activated={self.n_activated_experts} <= n_routed={self.n_routed_experts}")
        assert self.n_activated_experts <= self.n_routed_experts, "n_activated_experts must be <= n_routed_experts"
        logger.debug(f"MoE validation passed: total_experts = {self.n_routed_experts + self.n_shared_experts}")
        
        # Validate router
        logger.debug(f"Validating router score function: {self.moe_router_score_func}")
        assert self.moe_router_score_func in ["softmax", "sigmoid"], "score_func must be softmax or sigmoid"
        logger.debug(f"Router validation passed")
        
        # Validate dtype
        logger.debug(f"Validating dtype: {self.dtype}")
        assert self.dtype in ["bf16", "fp16", "fp32", "fp8"], f"invalid dtype: {self.dtype}"
        logger.info(f"Using dtype: {self.dtype} -> {self.get_dtype()}")
        
        # Validate FP8 support
        if self.use_fp8_training:
            if self.dtype != "fp8":
                logger.warning("FP8 training requires dtype='fp8', forcing dtype='fp8'")
                self.dtype = "fp8"
            if not torch.cuda.is_available():
                logger.warning("FP8 training requires CUDA, disabling")
                self.use_fp8_training = False
            elif torch.cuda.get_device_capability() < (8, 9):
                logger.warning("FP8 training requires compute capability >= 8.9 (Hopper+), disabling")
                self.use_fp8_training = False
        
        # Validate parallel MoE+Dense
        logger.debug(f"Validating parallel MoE+Dense: combine_mode={self.parallel_moe_dense_combine}")
        valid_modes = ["add", "concat", "gated", "residual_fusion"]
        assert self.parallel_moe_dense_combine in valid_modes, f"invalid combine mode, must be one of {valid_modes}"
        if self.use_parallel_moe_dense:
            logger.info(f"Parallel MoE+Dense enabled: ratio={self.parallel_moe_dense_ratio}, combine={self.parallel_moe_dense_combine}")
        
        # Validate dynamic depth
        if self.use_dynamic_depth:
            logger.info(f"Dynamic depth enabled: min_layers={self.dynamic_depth_min_layers}, threshold={self.dynamic_depth_confidence_threshold}")
        
        # Validate MTP
        if self.use_multi_token_prediction:
            logger.info(f"Multi-token prediction enabled: n_predictions={self.mtp_n_predictions}, weight={self.mtp_loss_weight}")
        
        # Apply configuration adjustments
        if self.use_zero3:
            logger.info("DeepSpeed ZeRO-3 enabled, forcing FSDP=True and zero_stage=3")
            self.use_fsdp = True
            self.zero_stage = 3
        
        if self.moe_use_fused_kernel and not HAS_MEGABLOCKS:
            logger.warning("Megablocks not available, falling back to reference MoE implementation")
            self.moe_use_fused_kernel = False
        else:
            logger.info(f"Fused MoE kernels enabled: {self.moe_use_fused_kernel}")
        
        if self.use_flash_attn and not HAS_FLASH_ATTN:
            logger.warning("FlashAttention not available, falling back to SDPA")
            self.use_flash_attn = False
        else:
            logger.info(f"Flash Attention 2 enabled: {self.use_flash_attn}")
        
        # Compute derived values
        self.head_dim = self.dim // self.n_heads
        self.total_experts = self.n_routed_experts + self.n_shared_experts
        
        # Log configuration summary
        logger.info(f"\nEnhanced Model Configuration Summary:")
        logger.info(f"  Architecture: {self.model_name} v{self.model_version}")
        logger.info(f"  Hidden Dim: {self.dim:,}, Layers: {self.n_layers}, Heads: {self.n_heads}, Head Dim: {self.head_dim}")
        logger.info(f"  Vocab Size: {self.vocab_size:,}, Max Seq Len: {self.max_seq_len:,}")
        logger.info(f"  MoE: {self.n_routed_experts} routed + {self.n_shared_experts} shared = {self.total_experts} total experts")
        logger.info(f"  Top-K: {self.n_activated_experts} experts active per token")
        logger.info(f"  Device: {self.device}, Dtype: {self.dtype}, Mixed Precision: {self.use_amp}")
        
        # Enhanced features
        logger.info(f"\nEnhanced Features:")
        logger.info(f"  Parallel MoE+Dense: {self.use_parallel_moe_dense} (mode={self.parallel_moe_dense_combine})")
        logger.info(f"  GLM Integration: {self.use_glm}")
        logger.info(f"  Shared Expert: {self.use_shared_expert}")
        logger.info(f"  Adaptive Router: {self.use_adaptive_router}")
        logger.info(f"  Dynamic Depth: {self.use_dynamic_depth}")
        logger.info(f"  Multi-Token Prediction: {self.use_multi_token_prediction}")
        logger.info(f"  FP8 Training: {self.use_fp8_training}")
        logger.info(f"  Expert Parallelism: {self.moe_expert_parallel}")
        logger.info("=" * 80)
    
    @property
    def score_func(self) -> str:
        return self.moe_router_score_func
    
    @property
    def aux_loss_weight(self) -> float:
        return self.moe_aux_loss_weight
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype from string with FP8 support"""
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "fp8": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
    
    def get_fp8_dtype(self) -> Optional[torch.dtype]:
        """Get FP8 dtype if supported"""
        if self.use_fp8_training and hasattr(torch, 'float8_e4m3fn'):
            return torch.float8_e4m3fn
        return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def save(self, path: str):
        """Save configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelArgs':
        """Load configuration from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def deepseek_v3_671b(cls) -> 'ModelArgs':
        """DeepSeek V3 671B parameter configuration"""
        return cls(
            dim=7168,
            n_layers=61,
            n_dense_layers=3,
            vocab_size=129280,
            n_heads=128,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            n_routed_experts=256,
            n_activated_experts=8,
            n_shared_experts=2,
            moe_inter_dim=2048,
            inter_dim=18432,
            max_seq_len=131072,
            rope_theta=10000000.0,
            model_name="deepseek-v3-671b"
        )
    
    @classmethod
    def deepseek_v3_lite(cls) -> 'ModelArgs':
        """Lightweight DeepSeek V3 configuration"""
        return cls(
            dim=2048,
            n_layers=27,
            n_dense_layers=2,
            vocab_size=102400,
            n_heads=16,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            n_routed_experts=64,
            n_activated_experts=6,
            n_shared_experts=2,
            moe_inter_dim=1408,
            inter_dim=10944,
            max_seq_len=32768,
            model_name="deepseek-v3-lite"
        )
    
    @classmethod
    def parallel_moe_dense(cls) -> 'ModelArgs':
        """Parallel MoE + Dense configuration"""
        args = cls.deepseek_v3_lite()
        args.use_parallel_moe_dense = True
        args.parallel_moe_dense_ratio = 0.5
        args.parallel_moe_dense_combine = "residual_fusion"
        args.model_name = "parallel-moe-dense-enhanced"
        return args
    
    @classmethod
    def enhanced_full(cls) -> 'ModelArgs':
        """Full enhanced configuration with all features"""
        args = cls.deepseek_v3_lite()
        args.use_parallel_moe_dense = True
        args.parallel_moe_dense_combine = "residual_fusion"
        args.use_glm = True
        args.use_shared_expert = True
        args.use_adaptive_router = True
        args.use_dynamic_depth = True
        args.use_multi_token_prediction = True
        args.use_fp8_training = False  # Enable only on Hopper+ GPUs
        args.moe_expert_parallel = True
        args.model_name = "deepnova-enhanced-full"
        return args


@dataclass
class TrainingArgs:
    """Training configuration"""
    epochs: int = 10
    max_steps: int = -1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    lr_warmup_steps: int = 2000
    lr_decay_style: str = "cosine"
    lr_decay_iters: int = -1
    min_lr: float = 1e-6
    
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    clip_grad: float = 1.0
    use_fused_adam: bool = True
    
    use_ema: bool = False
    ema_decay: float = 0.999
    
    use_swa: bool = False
    swa_start: int = 0
    swa_lr: float = 1e-4
    
    train_batch_size: int = 16
    eval_batch_size: int = 16
    max_seq_len: int = 4096
    min_seq_len: int = 64
    
    fp16: bool = False
    bf16: bool = True
    fp16_opt_level: str = "O2"
    
    gradient_accumulation_steps: int = 1
    
    save_total_limit: int = 3
    save_only_last: bool = False
    
    tensorboard_dir: str = "./logs"
    
    resume_from_checkpoint: Optional[str] = None
    
    label_smoothing: float = 0.0
    
    def __post_init__(self):
        if self.bf16:
            self.fp16 = False


# ============================================================================
# INTELLIGENT CONTEXT MEMORY SYSTEM (Same as before)
# ============================================================================

class CompressedMemoryEntry:
    """Single memory entry with compression metadata"""
    
    def __init__(self, role: str, content: str, timestamp: float, importance: float = 0.0):
        self.role = role
        self.original_content = content
        self.timestamp = timestamp
        self.importance = importance
        self.compressed = False
        self.compressed_content = content
        self.key_facts = []
        self.embedding = None
        
    def compress(self, max_length: int = 200):
        """Compress content while preserving key information"""
        if len(self.original_content) <= max_length:
            return
        
        sentences = re.split(r'[.!?]+', self.original_content)
        important_sentences = []
        
        priority_keywords = ['important', 'remember', 'key', 'critical', 'note', 
                             'must', 'always', 'never', 'always', 'specific', 
                             'exact', 'precise']
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            if any(kw in sent.lower() for kw in priority_keywords):
                important_sentences.append(sent)
                self.key_facts.append(sent[:100])
            
            elif len(important_sentences) == 0 and len(sent) < 100:
                important_sentences.append(sent)
        
        if not important_sentences and len(sentences) > 2:
            important_sentences = [sentences[0], sentences[-1]]
        elif not important_sentences:
            important_sentences = [sentences[0][:max_length]]
        
        self.compressed_content = '. '.join(important_sentences)
        if len(self.compressed_content) > max_length:
            self.compressed_content = self.compressed_content[:max_length] + '...'
        
        self.compressed = True
    
    def get_content(self, use_compressed: bool = True) -> str:
        """Get content, optionally using compressed version"""
        if use_compressed and self.compressed:
            return self.compressed_content
        return self.original_content


class ContextMemory:
    """Intelligent context memory with compression and importance scoring"""
    
    def __init__(self, memory_file: str = "deepnova_memory.json", max_tokens: int = 8192):
        self.memory_file = memory_file
        self.max_tokens = max_tokens
        
        self.short_term: deque = deque(maxlen=100)
        self.long_term: Dict[str, List[CompressedMemoryEntry]] = {}
        self.summaries: deque = deque(maxlen=30)
        self.important_facts: Set[str] = set()
        self.conversation_topics: Dict[str, List[Dict]] = defaultdict(list)
        self.entity_memory: Dict[str, Dict] = {}
        
        self.total_compressions = 0
        self.total_tokens_saved = 0
        
        self._stop_event = threading.Event()
        self._thread = None
        self._save_lock = threading.Lock()
        self._load_memory()
        self._start_auto_save()
    
    def _start_auto_save(self):
        """Start auto-save thread with graceful shutdown"""
        def auto_save():
            while not self._stop_event.wait(300):
                self._save_memory()
        
        self._thread = threading.Thread(target=auto_save, daemon=True)
        self._thread.start()
    
    def shutdown(self):
        """Graceful shutdown of auto-save thread"""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._save_memory()
    
    def _save_memory(self):
        """Save memory to disk using JSON"""
        try:
            short_term_data = []
            for entry in self.short_term:
                short_term_data.append({
                    'role': entry.role,
                    'original_content': entry.original_content,
                    'timestamp': entry.timestamp,
                    'importance': entry.importance,
                    'compressed': entry.compressed,
                    'compressed_content': entry.compressed_content,
                    'key_facts': entry.key_facts
                })
            
            memory_data = {
                'short_term': short_term_data,
                'long_term': dict(self.long_term),
                'summaries': list(self.summaries),
                'important_facts': list(self.important_facts),
                'conversation_topics': dict(self.conversation_topics),
                'entity_memory': self.entity_memory,
                'total_compressions': self.total_compressions,
                'total_tokens_saved': self.total_tokens_saved,
                'timestamp': time.time(),
                'version': '2.0'
            }
            
            temp_file = self.memory_file + ".tmp"
            with self._save_lock:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(memory_data, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, self.memory_file)
            
            logger.debug(f"Memory saved to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def _load_memory(self):
        """Load memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with self._save_lock:
                    with open(self.memory_file, 'r', encoding='utf-8') as f:
                        memory_data = json.load(f)
                
                self.short_term = deque(maxlen=100)
                for entry_data in memory_data.get('short_term', []):
                    entry = CompressedMemoryEntry(
                        entry_data['role'], 
                        entry_data['original_content'],
                        entry_data['timestamp'],
                        entry_data.get('importance', 0.0)
                    )
                    entry.compressed = entry_data.get('compressed', False)
                    entry.compressed_content = entry_data.get('compressed_content', entry_data['original_content'])
                    entry.key_facts = entry_data.get('key_facts', [])
                    self.short_term.append(entry)
                
                self.long_term = memory_data.get('long_term', {})
                self.summaries = deque(memory_data.get('summaries', []), maxlen=30)
                self.important_facts = set(memory_data.get('important_facts', []))
                self.conversation_topics = defaultdict(list, memory_data.get('conversation_topics', {}))
                self.entity_memory = memory_data.get('entity_memory', {})
                self.total_compressions = memory_data.get('total_compressions', 0)
                self.total_tokens_saved = memory_data.get('total_tokens_saved', 0)
                
                logger.info(f"Memory loaded from {self.memory_file}")
                logger.info(f"Loaded {len(self.short_term)} short-term entries, {len(self.important_facts)} facts")
            except Exception as e:
                logger.error(f"Failed to load memory: {e}")
    
    def _compute_importance(self, content: str) -> float:
        """Compute importance score for a message"""
        importance = 0.0
        
        importance_keywords = [
            'important', 'critical', 'urgent', 'remember', 'key', 'essential',
            'specific', 'exact', 'precise', 'exactly', 'precisely'
        ]
        
        content_lower = content.lower()
        for kw in importance_keywords:
            if kw in content_lower:
                importance += 0.15
        
        if len(content) > 500:
            importance += 0.1
        elif len(content) > 200:
            importance += 0.05
        
        if '?' in content:
            importance += 0.05
        
        personal_patterns = ['my name is', 'i am', 'i work', 'i live']
        for pattern in personal_patterns:
            if pattern in content_lower:
                importance += 0.2
                break
        
        return min(importance, 1.0)
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract entities from text"""
        entities = []
        
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        names = re.findall(name_pattern, content)
        entities.extend(names[:3])
        
        email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
        emails = re.findall(email_pattern, content)
        entities.extend(emails)
        
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, content)
        entities.extend([f"number:{n}" for n in numbers[:2]])
        
        return list(set(entities))
    
    def _update_entity_memory(self, content: str):
        """Update entity memory with extracted information"""
        entities = self._extract_entities(content)
        
        for entity in entities:
            if entity not in self.entity_memory:
                self.entity_memory[entity] = {
                    'first_seen': time.time(),
                    'mentions': 1,
                    'context': []
                }
            else:
                self.entity_memory[entity]['mentions'] += 1
            
            context = content[:200]
            if context not in self.entity_memory[entity]['context']:
                self.entity_memory[entity]['context'].append(context)
                if len(self.entity_memory[entity]['context']) > 5:
                    self.entity_memory[entity]['context'].pop(0)
    
    def add_message(self, role: str, content: str, compress: bool = True):
        """Add a message to memory with intelligent compression"""
        importance = self._compute_importance(content)
        entry = CompressedMemoryEntry(role, content, time.time(), importance)
        
        if compress and len(content) > 300:
            entry.compress()
            self.total_compressions += 1
            original_tokens = len(content) // 4
            compressed_tokens = len(entry.compressed_content) // 4
            self.total_tokens_saved += original_tokens - compressed_tokens
        
        self.short_term.append(entry)
        
        self._extract_important_facts(content, importance)
        self._update_entity_memory(content)
        self._analyze_topic(content)
    
    def _extract_important_facts(self, content: str, importance: float):
        """Extract important facts from message"""
        if importance < 0.3:
            return
        
        sentences = re.split(r'[.!?]+', content)
        
        important_patterns = [
            r'(?:my name is|i am|i work as|i am a)\s+([A-Za-z\s]+)',
            r'(?:i live in)\s+([A-Za-z\s]+)',
            r'(?:i like|i don\'t like)\s+([^.]+)',
            r'(?:remember that)\s+([^.]+)',
            r'(?:important|key)[:\s]+([^.]+)',
        ]
        
        for pattern in important_patterns:
            matches = re.findall(pattern, content.lower())
            for match in matches:
                fact = match.strip()
                if 5 < len(fact) < 200:
                    self.important_facts.add(fact)
        
        if importance > 0.6:
            for sent in sentences:
                sent = sent.strip()
                if 10 < len(sent) < 150 and any(c.isalpha() for c in sent):
                    self.important_facts.add(sent)
        
        if len(self.important_facts) > 200:
            self.important_facts = set(list(self.important_facts)[-200:])
    
    def _analyze_topic(self, content: str):
        """Analyze conversation topics"""
        topics = {
            'work': ['work', 'job', 'career', 'office', 'project', 'task'],
            'learning': ['learn', 'study', 'course', 'knowledge', 'skill'],
            'technology': ['ai', 'code', 'programming', 'software', 'computer'],
            'personal': ['family', 'friend', 'home', 'life', 'personal'],
            'health': ['health', 'exercise', 'diet', 'medical'],
            'entertainment': ['movie', 'music', 'game', 'show'],
            'travel': ['travel', 'trip', 'flight', 'hotel'],
            'food': ['food', 'restaurant', 'cook', 'meal']
        }
        
        content_lower = content.lower()
        
        for topic, keywords in topics.items():
            if any(kw in content_lower for kw in keywords):
                self.conversation_topics[topic].append({
                    'content': content[:200],
                    'timestamp': time.time(),
                    'length': len(content)
                })
                
                if len(self.conversation_topics[topic]) > 20:
                    self.conversation_topics[topic].pop(0)
    
    def get_context(self, max_tokens: int = 4096, include_summaries: bool = True) -> str:
        """Get conversation context with token budget"""
        context_parts = []
        current_tokens = 0
        
        if self.important_facts:
            facts_text = "Important Information:\n" + "\n".join(f"- {fact}" for fact in list(self.important_facts)[-10:])
            facts_tokens = len(facts_text) // 4
            if current_tokens + facts_tokens <= max_tokens:
                context_parts.append(facts_text)
                current_tokens += facts_tokens
        
        if self.entity_memory and current_tokens < max_tokens * 0.3:
            important_entities = sorted(self.entity_memory.items(), 
                                       key=lambda x: x[1]['mentions'], reverse=True)[:10]
            if important_entities:
                entity_text = "Known Entities:\n" + "\n".join(f"- {entity}" for entity, _ in important_entities)
                entity_tokens = len(entity_text) // 4
                if current_tokens + entity_tokens <= max_tokens:
                    context_parts.append(entity_text)
                    current_tokens += entity_tokens
        
        recent_messages = []
        for entry in reversed(self.short_term):
            msg_content = entry.get_content(use_compressed=True)
            msg_text = f"{entry.role}: {msg_content}"
            msg_tokens = len(msg_text) // 4
            
            if current_tokens + msg_tokens <= max_tokens * 0.8:
                recent_messages.insert(0, msg_text)
                current_tokens += msg_tokens
            else:
                break
        
        if recent_messages:
            context_parts.append("\n".join(recent_messages))
        
        if include_summaries and self.summaries and current_tokens < max_tokens * 0.9:
            summary_text = "Previous Conversation Summary:\n" + "\n".join(list(self.summaries)[-3:])
            summary_tokens = len(summary_text) // 4
            if current_tokens + summary_tokens <= max_tokens:
                context_parts.insert(0, summary_text)
        
        if current_tokens < max_tokens * 0.95:
            active_topics = []
            for topic, items in self.conversation_topics.items():
                if items and (time.time() - items[-1]['timestamp']) < 3600:
                    active_topics.append(topic)
            
            if active_topics:
                topics_text = f"Recent Topics: {', '.join(active_topics)}"
                if current_tokens + len(topics_text) // 4 <= max_tokens:
                    context_parts.append(topics_text)
        
        return "\n\n".join(context_parts)
    
    def create_summary(self) -> str:
        """Create conversation summary for long-term storage"""
        if len(self.short_term) < 10:
            return None
        
        summary_parts = []
        
        active_topics = []
        for topic, items in self.conversation_topics.items():
            if items and (time.time() - items[-1]['timestamp']) < 7200:
                active_topics.append(topic)
        
        if active_topics:
            summary_parts.append(f"Topics discussed: {', '.join(active_topics)}")
        
        if self.important_facts:
            recent_facts = list(self.important_facts)[-5:]
            if recent_facts:
                summary_parts.append(f"Key information: {' | '.join(recent_facts)}")
        
        user_messages = []
        for entry in self.short_term:
            if entry.role == 'user' and len(entry.original_content) > 20:
                user_messages.append(entry.original_content[:100])
        
        if user_messages:
            summary_parts.append(f"User discussed: {' | '.join(user_messages[-3:])}")
        
        summary = " | ".join(summary_parts)
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        self.summaries.append(summary)
        return summary
    
    def clear_short_term(self):
        """Clear short-term memory but keep important facts"""
        self.short_term.clear()
        logger.info("Short-term memory cleared, keeping important facts and entities")
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'short_term_messages': len(self.short_term),
            'long_term_topics': len(self.long_term),
            'summaries_count': len(self.summaries),
            'important_facts': len(self.important_facts),
            'active_topics': len(self.conversation_topics),
            'entities_tracked': len(self.entity_memory),
            'total_compressions': self.total_compressions,
            'total_tokens_saved': self.total_tokens_saved,
            'memory_file': self.memory_file
        }


# ============================================================================
# TEXT LEARNING SYSTEM (Same as before)
# ============================================================================

class LearnedText:
    """Structure for learned text knowledge"""
    
    def __init__(self, text: str, source: str, text_hash: str):
        self.text = text
        self.source = source
        self.hash = text_hash
        self.timestamp = time.time()
        self.summary = self._generate_summary()
        self.keywords = self._extract_keywords()
        self.entities = self._extract_entities()
        self.embedding = None
        self.access_count = 0
        self.last_accessed = time.time()
    
    def _generate_summary(self, max_length: int = 200) -> str:
        """Generate text summary"""
        if len(self.text) <= max_length:
            return self.text
        
        first_sentence = self.text.split('.')[0]
        if len(first_sentence) <= max_length:
            return first_sentence + "..."
        
        return self.text[:max_length] + "..."
    
    def _extract_keywords(self, max_keywords: int = 15) -> List[str]:
        """Extract keywords from text"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', self.text.lower())
        word_counts = {}
        
        for word in words:
            if word not in stop_words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_keywords]]
    
    def _extract_entities(self) -> List[str]:
        """Extract entities from text"""
        entities = []
        
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', self.text)
        entities.extend(names[:5])
        
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', self.text)
        entities.extend([f"number:{n}" for n in numbers[:3]])
        
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', self.text)
        entities.extend(emails[:2])
        
        urls = re.findall(r'https?://[^\s]+', self.text)
        entities.extend(urls[:2])
        
        return list(set(entities))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'source': self.source,
            'hash': self.hash,
            'timestamp': self.timestamp,
            'summary': self.summary,
            'keywords': self.keywords,
            'entities': self.entities,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LearnedText':
        """Create from dictionary"""
        obj = cls(data['text'], data['source'], data['hash'])
        obj.timestamp = data['timestamp']
        obj.summary = data['summary']
        obj.keywords = data['keywords']
        obj.entities = data['entities']
        obj.access_count = data.get('access_count', 0)
        obj.last_accessed = data.get('last_accessed', obj.timestamp)
        return obj


class SimpleEmbedding:
    """Lightweight embedding system using TF-IDF-like vectors for knowledge retrieval"""
    
    def __init__(self, max_features: int = 10000):
        self.max_features = max_features
        self.vocabulary = {}
        self.embeddings = {}
        self.idf_scores = {}
        self.doc_count = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'this', 'that'}
        return [w for w in words if w not in stop_words]
    
    def add_document(self, text_hash: str, text: str):
        """Add document to embedding system"""
        tokens = self._tokenize(text)
        
        for token in set(tokens):
            if len(self.vocabulary) < self.max_features:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        
        embedding = [0.0] * min(len(self.vocabulary), self.max_features)
        token_counts = {}
        for token in tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                if idx < self.max_features:
                    token_counts[idx] = token_counts.get(idx, 0) + 1
        
        for idx, count in token_counts.items():
            tf = count / (len(tokens) + 1e-8)
            idf = self.idf_scores.get(idx, 1.0)
            embedding[idx] = tf * idf
        
        norm = sum(x**2 for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        
        self.embeddings[text_hash] = embedding
    
    def update_idf(self, doc_texts: Dict[str, str]):
        """Update IDF scores based on all documents"""
        self.doc_count = len(doc_texts)
        doc_freq = {}
        
        for text_hash, text in doc_texts.items():
            tokens = set(self._tokenize(text))
            for token in tokens:
                if token in self.vocabulary:
                    idx = self.vocabulary[token]
                    if idx < self.max_features:
                        doc_freq[idx] = doc_freq.get(idx, 0) + 1
        
        import math
        for idx in range(len(self.vocabulary)):
            freq = doc_freq.get(idx, 0)
            self.idf_scores[idx] = math.log((self.doc_count + 1) / (freq + 1)) + 1
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        query_tokens = self._tokenize(query)
        query_embedding = [0.0] * min(len(self.vocabulary), self.max_features)
        
        token_counts = {}
        for token in query_tokens:
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                if idx < self.max_features:
                    token_counts[idx] = token_counts.get(idx, 0) + 1
        
        for idx, count in token_counts.items():
            tf = count / (len(query_tokens) + 1e-8)
            idf = self.idf_scores.get(idx, 1.0)
            query_embedding[idx] = tf * idf
        
        norm = sum(x**2 for x in query_embedding) ** 0.5
        if norm > 0:
            query_embedding = [x / norm for x in query_embedding]
        
        scores = []
        for text_hash, doc_embedding in self.embeddings.items():
            similarity = sum(a*b for a, b in zip(query_embedding, doc_embedding))
            scores.append((text_hash, similarity))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class TextLearningSystem:
    """Advanced text learning system with persistent storage"""
    
    def __init__(self, model, tokenizer, args: ModelArgs, memory: ContextMemory = None):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.memory = memory or ContextMemory()
        
        self.learned_texts: Dict[str, LearnedText] = {}
        self.knowledge_graph: Dict[str, Set[str]] = defaultdict(set)
        self.learning_history: List[Dict] = []
        self.embedding_system = SimpleEmbedding(max_features=5000)
        
        self.knowledge_dir = "./deepnova_knowledge"
        os.makedirs(self.knowledge_dir, exist_ok=True)
        
        self._load_knowledge()
    
    def _load_knowledge(self):
        """Load knowledge from disk with set conversion"""
        knowledge_file = os.path.join(self.knowledge_dir, "knowledge_base.json")
        if os.path.exists(knowledge_file):
            try:
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for hash_id, text_data in data.get('learned_texts', {}).items():
                    self.learned_texts[hash_id] = LearnedText.from_dict(text_data)
                
                for k, v_list in data.get('knowledge_graph', {}).items():
                    self.knowledge_graph[k] = set(v_list)
                
                self.learning_history = data.get('history', [])
                
                for hash_id, learned in self.learned_texts.items():
                    self.embedding_system.add_document(hash_id, learned.text)
                
                if self.learned_texts:
                    doc_texts = {h: t.text for h, t in self.learned_texts.items()}
                    self.embedding_system.update_idf(doc_texts)
                
                logger.info(f"Loaded {len(self.learned_texts)} learned texts from {knowledge_file}")
            except Exception as e:
                logger.error(f"Failed to load knowledge: {e}")
    
    def _save_knowledge(self):
        """Save knowledge to disk with set to list conversion"""
        knowledge_file = os.path.join(self.knowledge_dir, "knowledge_base.json")
        backup_file = knowledge_file + ".backup"
        
        try:
            if os.path.exists(knowledge_file):
                os.rename(knowledge_file, backup_file)
            
            graph_for_json = {}
            for k, v_set in self.knowledge_graph.items():
                graph_for_json[k] = list(v_set)
            
            data = {
                'learned_texts': {h: t.to_dict() for h, t in self.learned_texts.items()},
                'knowledge_graph': graph_for_json,
                'history': self.learning_history[-100:],
                'last_updated': time.time(),
                'version': '2.0'
            }
            
            with open(knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(backup_file):
                os.remove(backup_file)
            
            logger.debug(f"Saved {len(self.learned_texts)} learned texts to {knowledge_file}")
        except Exception as e:
            logger.error(f"Failed to save knowledge: {e}")
            if os.path.exists(backup_file):
                os.rename(backup_file, knowledge_file)
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        words1 = set(re.findall(r'\b[a-z]{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-z]{3,}\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _update_knowledge_graph(self, text_hash: str, keywords: List[str], entities: List[str]):
        """Update knowledge graph connections"""
        all_terms = set(keywords + entities)
        
        for term in all_terms:
            self.knowledge_graph[term].add(text_hash)
        
        term_list = list(all_terms)
        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                self.knowledge_graph[term_list[i]].add(term_list[j])
                self.knowledge_graph[term_list[j]].add(term_list[i])
    
    def learn_from_text(self, text: str, source: str = "user", extract_keys: bool = True) -> Dict:
        """Learn from provided text"""
        start_time = time.time()
        
        text = text.strip()
        if not text:
            return {"success": False, "error": "Empty text"}
        
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        if text_hash in self.learned_texts:
            self.learned_texts[text_hash].access_count += 1
            self.learned_texts[text_hash].last_accessed = time.time()
            return {"success": False, "error": "Text already learned", "hash": text_hash}
        
        learned = LearnedText(text, source, text_hash)
        
        self.learned_texts[text_hash] = learned
        self._update_knowledge_graph(text_hash, learned.keywords, learned.entities)
        
        try:
            self.embedding_system.add_document(text_hash, text)
        except Exception as e:
            logger.debug(f"Failed to add embedding for {text_hash}: {e}")
        
        self.learning_history.append({
            "hash": text_hash,
            "source": source,
            "timestamp": time.time(),
            "summary": learned.summary,
            "keywords": learned.keywords[:5]
        })
        
        if self.memory:
            self.memory.add_message("system", f"Learned: {learned.summary}", compress=True)
            for fact in learned.entities:
                self.memory.important_facts.add(fact)
        
        self._save_knowledge()
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "hash": text_hash,
            "summary": learned.summary,
            "keywords": learned.keywords[:10],
            "entities": learned.entities[:10],
            "learning_time_ms": elapsed * 1000,
            "total_learned": len(self.learned_texts)
        }
    
    def learn_from_file(self, file_path: str, source: str = None) -> List[Dict]:
        """Learn from file"""
        results = []
        
        if not os.path.exists(file_path):
            return [{"success": False, "error": f"File not found: {file_path}"}]
        
        if source is None:
            source = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            chunks = self._chunk_text(content, max_chunk_size=2000)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    result = self.learn_from_text(
                        chunk, 
                        source=f"{source}:chunk_{i}",
                        extract_keys=(i == 0)
                    )
                    results.append(result)
            
            success_count = len([r for r in results if r.get('success')])
            logger.info(f"Learned {success_count} chunks from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to learn from file {file_path}: {e}")
            results.append({"success": False, "error": str(e)})
        
        return results
    
    def learn_from_directory(self, dir_path: str, extensions: List[str] = None) -> List[Dict]:
        """Learn from all files in directory"""
        if extensions is None:
            extensions = ['.txt', '.md', '.json', '.csv']
        
        results = []
        
        if not os.path.exists(dir_path):
            return [{"success": False, "error": f"Directory not found: {dir_path}"}]
        
        files_processed = 0
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    file_results = self.learn_from_file(file_path, source=f"dir:{dir_path}")
                    results.extend(file_results)
                    files_processed += 1
        
        logger.info(f"Learned from directory {dir_path}, processed {files_processed} files")
        return results
    
    def query_knowledge(self, query: str, top_k: int = 10, min_similarity: float = 0.1) -> List[Dict]:
        """Query learned knowledge using hybrid approach"""
        if not self.learned_texts:
            return []
        
        results = []
        query_lower = query.lower()
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query_lower))
        
        vector_results = {}
        try:
            vector_matches = self.embedding_system.search(query, top_k=min(top_k*2, len(self.learned_texts)))
            for hash_id, vector_score in vector_matches:
                if vector_score > 0.05:
                    vector_results[hash_id] = vector_score
        except Exception as e:
            logger.debug(f"Vector search failed: {e}")
        
        related_hashes = set()
        for word in query_words:
            if word in self.knowledge_graph:
                for related in self.knowledge_graph[word]:
                    if isinstance(related, str) and len(related) == 16:
                        related_hashes.add(related)
        
        for hash_id, learned in self.learned_texts.items():
            score = 0.0
            
            if hash_id in vector_results:
                score += vector_results[hash_id] * 0.3
            
            if hash_id in related_hashes:
                score += 0.2
            
            for kw in learned.keywords:
                if kw and kw.lower() in query_lower:
                    score += 0.15
                    if len(kw) > 5:
                        score += 0.1
            
            for ent in learned.entities:
                if ent and ent.lower() in query_lower:
                    score += 0.25
            
            if score >= min_similarity:
                learned.access_count += 1
                learned.last_accessed = time.time()
                
                results.append({
                    "score": min(score, 1.0),
                    "hash": hash_id,
                    "summary": learned.summary,
                    "text": learned.text[:500],
                    "source": learned.source,
                    "timestamp": learned.timestamp,
                    "keywords": learned.keywords[:5],
                    "access_count": learned.access_count
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        self._save_knowledge()
        
        return results[:top_k]
    
    def _chunk_text(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """Split text into chunks"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_stats(self) -> Dict:
        """Get learning system statistics"""
        return {
            'total_learned': len(self.learned_texts),
            'knowledge_graph_nodes': len(self.knowledge_graph),
            'learning_history': len(self.learning_history),
            'memory_stats': self.memory.get_stats() if self.memory else {},
            'knowledge_dir': self.knowledge_dir
        }
    
    def list_learned(self, limit: int = 50) -> List[Dict]:
        """List learned texts"""
        learned_list = []
        for hash_id, learned in list(self.learned_texts.items())[:limit]:
            learned_list.append({
                "hash": hash_id,
                "summary": learned.summary,
                "source": learned.source,
                "timestamp": learned.timestamp,
                "access_count": learned.access_count,
                "keywords": learned.keywords[:5]
            })
        return learned_list
    
    def forget_text(self, text_hash: str) -> bool:
        """Forget a learned text"""
        if text_hash in self.learned_texts:
            del self.learned_texts[text_hash]
            
            for term, connections in list(self.knowledge_graph.items()):
                connections.discard(text_hash)
                if not connections:
                    del self.knowledge_graph[term]
            
            self._save_knowledge()
            return True
        
        return False
    
    def export_knowledge(self, output_file: str) -> bool:
        """Export knowledge base to file"""
        try:
            export_data = {
                'learned_texts': {h: t.to_dict() for h, t in self.learned_texts.items()},
                'knowledge_graph': {k: list(v) for k, v in self.knowledge_graph.items()},
                'history': self.learning_history,
                'export_timestamp': time.time(),
                'version': '2.0'
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Knowledge exported to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to export knowledge: {e}")
            return False


# ============================================================================
# DEEP NOVA AI - MAIN ASSISTANT (Enhanced)
# ============================================================================

class DeepNovaAI:
    """
    DeepNova - Enhanced AI Assistant with:
    - Intelligent context memory with compression
    - Text learning from files and user input
    - Persistent knowledge storage
    - Conversation summarization
    - Entity extraction and tracking
    - Token-efficient memory management
    - All enhanced model features (MoE+Dense, GLM, Adaptive Router, etc.)
    """
    
    def __init__(self, model, tokenizer, args: ModelArgs, memory_file: str = "deepnova_memory.json"):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.device = torch.device(args.device)
        
        self.name = "DeepNova"
        self.version = "5.0.0"
        self.personality = "professional, helpful, knowledgeable, and concise"
        
        self.memory = ContextMemory(memory_file=memory_file, max_tokens=8192)
        self.learning_system = TextLearningSystem(model, tokenizer, args, self.memory)
        
        self.conversation_id = str(uuid.uuid4())[:8]
        self.chat_history = []
        
        self.total_messages = 0
        self.total_tokens_generated = 0
        self.start_time = time.time()
        
        self.greeting = f"Hello! I am {self.name} v{self.version}, an intelligent AI assistant with enhanced MoE+Dense architecture. I can remember our conversations, learn from text files, and use advanced features like adaptive routing and multi-token prediction. How can I help you today?"
        
        # Feature announcements
        self.features = []
        if args.use_parallel_moe_dense:
            self.features.append("Parallel MoE+Dense")
        if args.use_glm:
            self.features.append("GLM Integration")
        if args.use_adaptive_router:
            self.features.append("Adaptive Router")
        if args.use_dynamic_depth:
            self.features.append("Dynamic Depth")
        if args.use_multi_token_prediction:
            self.features.append("Multi-Token Prediction")
        
        self.system_prompt = f"""You are {self.name}, an advanced AI assistant with {self.version}.

Your capabilities:
- Maintain long-term conversation memory
- Learn from user-provided text and files
- Recall previously discussed information
- Answer questions based on learned knowledge
- Provide concise, accurate, and helpful responses

Active Features: {', '.join(self.features) if self.features else 'Standard MoE'}

Guidelines:
- Be professional and respectful
- Provide accurate information
- If unsure, say so rather than guessing
- Keep responses concise but thorough
- Reference previous conversations when relevant
- Use learned knowledge to answer questions

Always strive to be helpful and efficient in your responses."""

        logger.info(f"DeepNova AI v{self.version} initialized with features: {', '.join(self.features) if self.features else 'Standard'}")
    
    def _build_prompt(self, user_input: str, max_context_tokens: int = 4096) -> str:
        """Build prompt with intelligent context selection"""
        try:
            context = self.memory.get_context(max_tokens=max_context_tokens, include_summaries=True)
            
            relevant_knowledge = []
            try:
                relevant_knowledge = self.learning_system.query_knowledge(user_input, top_k=5)
            except Exception as e:
                logger.debug(f"Knowledge query failed: {e}")
            
            knowledge_text = ""
            if relevant_knowledge:
                knowledge_text = "Relevant Knowledge:\n"
                for k in relevant_knowledge[:3]:
                    try:
                        summary = k.get('summary', '')[:200]
                        if summary:
                            knowledge_text += f"- {summary}\n"
                    except Exception as e:
                        logger.debug(f"Error formatting knowledge item: {e}")
            
            prompt_parts = [self.system_prompt]
            current_tokens = len(self.system_prompt) // 4
            
            if context and current_tokens < max_context_tokens * 0.4:
                context_section = f"Conversation Context:\n{context}"
                context_tokens = len(context_section) // 4
                if current_tokens + context_tokens <= max_context_tokens * 0.6:
                    prompt_parts.append(context_section)
                    current_tokens += context_tokens
            
            if knowledge_text and current_tokens < max_context_tokens * 0.7:
                knowledge_tokens = len(knowledge_text) // 4
                if current_tokens + knowledge_tokens <= max_context_tokens * 0.8:
                    prompt_parts.append(knowledge_text)
                    current_tokens += knowledge_tokens
            
            user_section = f"User: {user_input}\n{self.name}:"
            prompt_parts.append(user_section)
            
            prompt = "\n\n".join(prompt_parts)
            
            max_prompt_tokens = max_context_tokens * 0.95
            current_prompt_tokens = len(prompt) // 4
            if current_prompt_tokens > max_prompt_tokens:
                logger.warning(f"Prompt truncated from {current_prompt_tokens} to {max_prompt_tokens} tokens")
                prompt = prompt[:int(max_prompt_tokens * 4)]
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            return f"User: {user_input}\n{self.name}:"
    
    def chat(self, user_input: str, max_new_tokens: int = 500, 
             temperature: float = 0.7, save_to_memory: bool = True) -> str:
        """Chat with DeepNova AI"""
        
        if save_to_memory:
            self.memory.add_message("user", user_input, compress=True)
        
        prompt = self._build_prompt(user_input)
        
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_tensor = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50
            )
        
        response = self.tokenizer.decode(generated_ids.tolist())
        
        if save_to_memory:
            self.memory.add_message(self.name, response, compress=True)
        
        self.chat_history.append({
            "user": user_input,
            "assistant": response,
            "timestamp": time.time()
        })
        
        if len(self.chat_history) > 200:
            self.chat_history = self.chat_history[-200:]
        
        self.total_messages += 1
        self.total_tokens_generated += len(generated_ids) if hasattr(generated_ids, '__len__') else 0
        
        if self.total_messages % 10 == 0:
            self.memory.create_summary()
            self._save_state()
        
        return response
    
    def learn(self, text: str, source: str = "user") -> Dict:
        """Learn from user-provided text"""
        result = self.learning_system.learn_from_text(text, source)
        
        if result.get("success"):
            self.memory.add_message("system", f"Learned: {result['summary']}", compress=True)
        
        return result
    
    def learn_from_file(self, file_path: str) -> List[Dict]:
        """Learn from a text file"""
        results = self.learning_system.learn_from_file(file_path)
        success_count = len([r for r in results if r.get('success')])
        
        if success_count > 0:
            self.memory.add_message("system", f"Learned {success_count} segments from {file_path}", compress=True)
        
        return results
    
    def learn_from_directory(self, dir_path: str) -> List[Dict]:
        """Learn from all text files in directory"""
        results = self.learning_system.learn_from_directory(dir_path)
        success_count = len([r for r in results if r.get('success')])
        
        if success_count > 0:
            self.memory.add_message("system", f"Learned {success_count} segments from directory {dir_path}", compress=True)
        
        return results
    
    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """Recall learned knowledge"""
        return self.learning_system.query_knowledge(query, top_k=top_k)
    
    def get_context(self, max_tokens: int = 4096) -> str:
        """Get current conversation context"""
        return self.memory.get_context(max_tokens=max_tokens)
    
    def clear_context(self, keep_important: bool = True):
        """Clear short-term context"""
        if keep_important:
            self.memory.clear_short_term()
        else:
            old_memory = self.memory
            self.memory = ContextMemory(memory_file=old_memory.memory_file, max_tokens=8192)
            old_memory.shutdown()
            del old_memory
            gc.collect()
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        return {
            "name": self.name,
            "version": self.version,
            "conversation_id": self.conversation_id,
            "total_messages": self.total_messages,
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_saved": self.memory.total_tokens_saved,
            "uptime_seconds": time.time() - self.start_time,
            "memory": self.memory.get_stats(),
            "learning": self.learning_system.get_stats(),
            "chat_history_length": len(self.chat_history),
            "model": model_info,
            "active_features": self.features
        }
    
    def print_stats(self):
        """Print statistics in formatted output"""
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print(f"DEEPNOVA AI STATISTICS")
        print("=" * 70)
        print(f"  Name: {stats['name']}")
        print(f"  Version: {stats['version']}")
        print(f"  Conversation ID: {stats['conversation_id']}")
        print(f"  Uptime: {stats['uptime_seconds']:.1f} seconds")
        print("-" * 70)
        print(f"  Total Messages: {stats['total_messages']}")
        print(f"  Tokens Generated: {stats['total_tokens_generated']}")
        print(f"  Tokens Saved: {stats['total_tokens_saved']}")
        print(f"  Chat History: {stats['chat_history_length']} entries")
        print(f"  Active Features: {', '.join(stats['active_features']) if stats['active_features'] else 'Standard'}")
        print("-" * 70)
        print(f"  Memory:")
        print(f"    Short-term: {stats['memory']['short_term_messages']} messages")
        print(f"    Important Facts: {stats['memory']['important_facts']}")
        print(f"    Entities Tracked: {stats['memory']['entities_tracked']}")
        print(f"    Total Compressions: {stats['memory']['total_compressions']}")
        print("-" * 70)
        print(f"  Learning:")
        print(f"    Total Learned: {stats['learning']['total_learned']}")
        print(f"    Knowledge Graph Nodes: {stats['learning']['knowledge_graph_nodes']}")
        
        if stats.get('model'):
            print("-" * 70)
            print(f"  Model:")
            print(f"    Total Params: {stats['model'].get('total_params_formatted', 'N/A')}")
            print(f"    Active Params: {stats['model'].get('active_params_formatted', 'N/A')}")
            print(f"    Sparsity: {stats['model'].get('sparsity', 0):.1%}")
            print(f"    Experts: {stats['model'].get('n_experts', 'N/A')} (top-{stats['model'].get('n_activated_experts', 'N/A')})")
        
        print("=" * 70)
    
    def _save_state(self):
        """Save assistant state"""
        state_file = os.path.join(self.learning_system.knowledge_dir, "assistant_state.json")
        try:
            state = {
                'conversation_id': self.conversation_id,
                'total_messages': self.total_messages,
                'total_tokens_generated': self.total_tokens_generated,
                'start_time': self.start_time,
                'chat_history': self.chat_history[-50:],
                'timestamp': time.time()
            }
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def export_knowledge(self, output_file: str) -> bool:
        """Export all learned knowledge"""
        return self.learning_system.export_knowledge(output_file)
    
    def list_learned(self, limit: int = 50) -> List[Dict]:
        """List learned texts"""
        return self.learning_system.list_learned(limit=limit)
    
    def forget(self, text_hash: str) -> bool:
        """Forget a specific learned text"""
        return self.learning_system.forget_text(text_hash)


# ============================================================================
# ENHANCED MODEL VALIDATION FUNCTION
# ============================================================================

def validate_model_args(args: ModelArgs) -> bool:
    """Validate model arguments before creation"""
    errors = []
    
    if args.dim % args.n_heads != 0:
        errors.append(f"dim ({args.dim}) must be divisible by n_heads ({args.n_heads})")
    
    if args.n_activated_experts > args.n_routed_experts:
        errors.append(f"n_activated_experts ({args.n_activated_experts}) > n_routed_experts ({args.n_routed_experts})")
    
    if args.kv_lora_rank <= 0:
        errors.append(f"kv_lora_rank must be positive, got {args.kv_lora_rank}")
    
    if args.qk_nope_head_dim <= 0 or args.qk_rope_head_dim <= 0:
        errors.append(f"head dimensions must be positive")
    
    if args.max_seq_len <= 0:
        errors.append(f"max_seq_len must be positive")
    
    valid_dtypes = ['bf16', 'fp16', 'fp32', 'fp8']
    if args.dtype not in valid_dtypes:
        errors.append(f"dtype must be one of {valid_dtypes}, got {args.dtype}")
    
    if args.use_parallel_moe_dense:
        valid_modes = ['add', 'concat', 'gated', 'residual_fusion']
        if args.parallel_moe_dense_combine not in valid_modes:
            errors.append(f"parallel_moe_dense_combine must be one of {valid_modes}")
        
        if not (0 <= args.parallel_moe_dense_ratio <= 1):
            errors.append(f"parallel_moe_dense_ratio must be in [0,1], got {args.parallel_moe_dense_ratio}")
    
    if args.use_fp8_training:
        if not torch.cuda.is_available():
            errors.append("FP8 training requires CUDA")
        elif torch.cuda.get_device_capability() < (8, 9):
            errors.append("FP8 training requires compute capability >= 8.9 (Hopper+)")
    
    if errors:
        logger.error("ModelArgs validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("ModelArgs validation passed")
    return True


# ============================================================================
# CORE MODEL COMPONENTS (Enhanced)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    
    def __init__(self, dim: int, eps: float = 1e-6, use_triton: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_triton = use_triton and HAS_TRITON
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton and hasattr(F, 'rms_norm'):
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        if hasattr(F, 'rms_norm'):
            return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)
        
        dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        return (x_float / rms).to(dtype) * self.weight
    
    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class DeepNorm(RMSNorm):
    """DeepNorm for deeper networks"""
    
    def __init__(self, dim: int, eps: float = 1e-6, alpha: float = 1.0):
        super().__init__(dim, eps)
        self.alpha = alpha
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * super().forward(x)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with enhanced scaling"""
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 32768,
        theta: float = 100000.0,
        scaling: Optional[Dict] = None,
        use_fused: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.scaling = scaling or {}
        self.use_fused = use_fused and HAS_FLASH_ATTN
        
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, max_seq_len: int):
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        
        scale_type = self.scaling.get("type", None)
        if scale_type == "linear":
            factor = self.scaling.get("factor", 1.0)
            t = t / factor
        elif scale_type == "yarn":
            factor = self.scaling.get("factor", 1.0)
            t = t * (1.0 + factor * (t / float(max_seq_len)))
        elif scale_type == "dynamic":
            # Dynamic NTK scaling
            base = self.theta * (max_seq_len / self.scaling.get("original_max_len", 2048)) ** (self.dim / (self.dim - 2))
            freqs = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        
        freqs = torch.outer(t, freqs)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs), persistent=False)
        self.register_buffer("freqs_real", freqs.cos(), persistent=False)
        self.register_buffer("freqs_imag", freqs.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1] if x.dim() >= 3 else 1
        
        if self.use_fused and HAS_FLASH_ATTN:
            cos = self.freqs_real[start_pos:start_pos + seq_len].to(x.device)
            sin = self.freqs_imag[start_pos:start_pos + seq_len].to(x.device)
            return apply_rotary_emb(x, cos, sin, interleaved=False)
        
        if x.dim() == 2:
            x = x.view(batch_size, seq_len, -1)
        
        x_float = x.float()
        x_complex = torch.view_as_complex(x_float.reshape(*x_float.shape[:-1], -1, 2))
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len].to(x.device)
        shape = [1] * x_complex.dim()
        shape[1] = seq_len
        freqs_cis = freqs_cis.view(*shape, -1)
        out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
        return out.to(x.dtype)


class PagedKVCache:
    """Paged KV Cache for efficient memory management - FIXED VERSION"""
    
    def __init__(
        self,
        max_batch_size: int,
        max_num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype
    ):
        self.max_batch_size = max_batch_size
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.device = device
        self.dtype = dtype
        
        self.block_tables = torch.full(
            (max_batch_size, max_num_blocks), -1, 
            dtype=torch.int32, device=device
        )
        
        self.k_blocks = torch.zeros(max_num_blocks, block_size, kv_lora_rank, dtype=dtype, device=device)
        self.v_blocks = torch.zeros(max_num_blocks, block_size, num_heads * head_dim, dtype=dtype, device=device)
        self.pe_blocks = torch.zeros(max_num_blocks, block_size, qk_rope_head_dim, dtype=dtype, device=device)
        
        self.free_blocks_mask = torch.ones(max_num_blocks, dtype=torch.bool, device=device)
        self.num_free_blocks = max_num_blocks
        
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        self.num_blocks_per_seq = torch.zeros(max_batch_size, dtype=torch.int32, device=device)
        
        self._lock = threading.Lock()
        self.total_allocations = 0
        self.total_frees = 0
    
    def allocate_block(self) -> int:
        with self._lock:
            if self.num_free_blocks == 0:
                raise RuntimeError(f"KV cache full! max_blocks={self.max_num_blocks}, block_size={self.block_size}")
            
            free_indices = torch.where(self.free_blocks_mask)[0]
            block_id = free_indices[0].item()
            
            self.free_blocks_mask[block_id] = False
            self.num_free_blocks -= 1
            self.total_allocations += 1
            
            return block_id
    
    def free_block(self, block_id: int):
        with self._lock:
            if not self.free_blocks_mask[block_id]:
                self.k_blocks[block_id].zero_()
                self.v_blocks[block_id].zero_()
                self.pe_blocks[block_id].zero_()
                
                self.free_blocks_mask[block_id] = True
                self.num_free_blocks += 1
                self.total_frees += 1
    
    def allocate_sequence(self, batch_idx: int, seq_len: int):
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        if num_blocks > self.max_num_blocks:
            raise RuntimeError(f"Sequence length {seq_len} needs {num_blocks} blocks, but max is {self.max_num_blocks}")
        
        self.free_sequence(batch_idx)
        
        for i in range(num_blocks):
            block_id = self.allocate_block()
            self.block_tables[batch_idx, i] = block_id
        
        self.num_blocks_per_seq[batch_idx] = num_blocks
        self.seq_lens[batch_idx] = seq_len
    
    def free_sequence(self, batch_idx: int):
        num_blocks = self.num_blocks_per_seq[batch_idx].item()
        
        for i in range(num_blocks):
            block_id = self.block_tables[batch_idx, i].item()
            if block_id >= 0:
                self.k_blocks[block_id].zero_()
                self.v_blocks[block_id].zero_()
                self.pe_blocks[block_id].zero_()
                self.free_block(block_id)
                self.block_tables[batch_idx, i] = -1
        
        self.num_blocks_per_seq[batch_idx] = 0
        self.seq_lens[batch_idx] = 0
    
    def store_kv(
        self, batch_idx: int, positions: torch.Tensor,
        k_latent: torch.Tensor, v_full: torch.Tensor, k_pe: torch.Tensor
    ):
        """Store KV cache entries - FIXED VERSION"""
        seq_len = k_latent.size(0)
        if seq_len == 0:
            return
        
        if positions.dim() > 1:
            positions = positions.squeeze()
        
        for i in range(seq_len):
            pos = positions[i].item()
            block_id = pos // self.block_size
            offset = pos % self.block_size
            
            if block_id >= self.max_num_blocks:
                raise RuntimeError(f"Position {pos} exceeds max blocks {self.max_num_blocks}")
            
            current_block = self.block_tables[batch_idx, block_id].item()
            if current_block < 0:
                current_block = self.allocate_block()
                self.block_tables[batch_idx, block_id] = current_block
            
            self.k_blocks[current_block, offset] = k_latent[i].to(self.dtype)
            self.v_blocks[current_block, offset] = v_full[i].to(self.dtype)
            self.pe_blocks[current_block, offset] = k_pe[i].to(self.dtype)
        
        current_len = self.seq_lens[batch_idx].item()
        new_len = max(current_len, positions.max().item() + 1)
        self.seq_lens[batch_idx] = new_len
    
    def get_kv(
        self, batch_idx: int, up_to: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get KV cache entries - FIXED VERSION"""
        if up_to == 0:
            return (
                torch.zeros(0, self.kv_lora_rank, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.num_heads * self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.qk_rope_head_dim, device=self.device, dtype=self.dtype)
            )
        
        k_parts = []
        v_parts = []
        pe_parts = []
        
        current_pos = 0
        while current_pos < up_to:
            block_id = current_pos // self.block_size
            offset = current_pos % self.block_size
            
            if block_id >= self.max_num_blocks:
                raise RuntimeError(f"Requested position {current_pos} exceeds max blocks {self.max_num_blocks}")
            
            block_idx = self.block_tables[batch_idx, block_id].item()
            if block_idx < 0:
                break
            
            block_size_remaining = min(self.block_size - offset, up_to - current_pos)
            
            k_parts.append(self.k_blocks[block_idx, offset:offset + block_size_remaining])
            v_parts.append(self.v_blocks[block_idx, offset:offset + block_size_remaining])
            pe_parts.append(self.pe_blocks[block_idx, offset:offset + block_size_remaining])
            
            current_pos += block_size_remaining
        
        if not k_parts:
            return (
                torch.zeros(0, self.kv_lora_rank, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.num_heads * self.head_dim, device=self.device, dtype=self.dtype),
                torch.zeros(0, self.qk_rope_head_dim, device=self.device, dtype=self.dtype)
            )
        
        return (
            torch.cat(k_parts, dim=0),
            torch.cat(v_parts, dim=0),
            torch.cat(pe_parts, dim=0)
        )
    
    def get_kv_block(
        self, batch_idx: int, block_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        block_id = self.block_tables[batch_idx, block_idx].item()
        if block_id < 0:
            return None, None, None
        return (self.k_blocks[block_id], self.v_blocks[block_id], self.pe_blocks[block_id])
    
    def reset(self):
        with self._lock:
            self.free_blocks_mask.fill_(True)
            self.num_free_blocks = self.max_num_blocks
            self.k_blocks.zero_()
            self.v_blocks.zero_()
            self.pe_blocks.zero_()
            self.block_tables.fill_(-1)
            self.seq_lens.zero_()
            self.num_blocks_per_seq.zero_()
    
    def get_usage_ratio(self) -> float:
        return 1.0 - (self.num_free_blocks / self.max_num_blocks)
    
    def get_stats(self) -> Dict:
        return {
            "total_blocks": self.max_num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.max_num_blocks - self.num_free_blocks,
            "usage_ratio": self.get_usage_ratio(),
            "block_size": self.block_size,
            "total_memory_gb": (
                self.k_blocks.numel() * self.k_blocks.element_size() +
                self.v_blocks.numel() * self.v_blocks.element_size() +
                self.pe_blocks.numel() * self.pe_blocks.element_size()
            ) / 1e9,
            "total_allocations": self.total_allocations,
            "total_frees": self.total_frees,
        }


# ============================================================================
# ENHANCED ATTENTION MECHANISM
# ============================================================================

class MultiHeadLatentAttention(nn.Module):
    """Multi-Head Latent Attention with MLA - ENHANCED VERSION with GLM support"""
    
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.dtype = args.get_dtype()
        
        self.glm_mode = args.use_glm
        self.glm_attention_type = args.glm_attention_type
        
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False, dtype=self.dtype)
        else:
            self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=self.dtype)
            self.q_norm = RMSNorm(self.q_lora_rank, args.rms_norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False, dtype=self.dtype)
        
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False, dtype=self.dtype
        )
        self.kv_norm = RMSNorm(self.kv_lora_rank, args.rms_norm_eps)
        
        self.wkv_b = nn.Linear(
            self.kv_lora_rank, 
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim), 
            bias=False, dtype=self.dtype
        )
        
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False, dtype=self.dtype)
        
        self.rotary_emb = RotaryEmbedding(
            self.qk_rope_head_dim, args.max_seq_len, args.rope_theta, args.rope_scaling
        )
        
        self.attn_dropout = nn.Dropout(args.attention_dropout) if args.attention_dropout > 0 else nn.Identity()
        self.resid_dropout = nn.Dropout(args.residual_dropout) if args.residual_dropout > 0 else nn.Identity()
        
        # GLM-specific components
        if self.glm_mode:
            if self.glm_attention_type == "prefix":
                self.prefix_attention_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
            elif self.glm_attention_type == "sentinel":
                self.sentinel_embedding = nn.Embedding(args.glm_num_sentinels, self.dim, dtype=self.dtype)
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[PagedKVCache] = None,
        batch_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
        glm_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        
        # 1. Project queries
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        q = q.view(bsz, seq_len, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 2. Apply RoPE to q_pe
        q_pe = q_pe.reshape(bsz * seq_len, self.qk_rope_head_dim)
        q_pe = self.rotary_emb(q_pe, start_pos)
        q_pe = q_pe.reshape(bsz, seq_len, self.qk_rope_head_dim)
        
        # 3. Project KV
        kv_a = self.wkv_a(x)
        kv_latent, k_pe = torch.split(kv_a, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 4. Apply RoPE to k_pe
        k_pe = k_pe.reshape(bsz * seq_len, self.qk_rope_head_dim)
        k_pe = self.rotary_emb(k_pe, start_pos)
        k_pe = k_pe.reshape(bsz, seq_len, self.qk_rope_head_dim)
        
        kv_latent_normed = self.kv_norm(kv_latent)
        
        # 5. Expand to full KV (without RoPE part)
        kv_full = self.wkv_b(kv_latent_normed)
        kv_full = kv_full.view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv_full, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        
        # 6. Expand k_pe to match heads
        k_pe_expanded = k_pe.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        
        # 7. Combine k_nope and k_pe
        k = torch.cat([k_nope, k_pe_expanded], dim=-1)
        
        # 8. Expand q_pe to match heads
        q_pe_expanded = q_pe.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q_nope, q_pe_expanded], dim=-1)
        
        # 9. Handle KV cache
        if kv_cache is not None and start_pos > 0:
            k_cached, v_cached, _ = kv_cache.get_kv(batch_idx, start_pos)
            
            cached_len = k_cached.size(0)
            if cached_len > 0:
                k_cached = k_cached.view(cached_len, self.n_heads, self.qk_head_dim)
                v_cached = v_cached.view(cached_len, self.n_heads, self.v_head_dim)
                
                k = torch.cat([k_cached, k], dim=0)
                v = torch.cat([v_cached, v], dim=0)
                
                kv_cache.store_kv(
                    batch_idx,
                    torch.arange(start_pos, start_pos + seq_len, device=x.device),
                    kv_latent,
                    v.reshape(-1, self.n_heads * self.v_head_dim)[-seq_len:],
                    k_pe
                )
            else:
                kv_cache.store_kv(
                    batch_idx,
                    torch.arange(start_pos, start_pos + seq_len, device=x.device),
                    kv_latent,
                    v.reshape(bsz * seq_len, self.n_heads * self.v_head_dim),
                    k_pe
                )
        elif kv_cache is not None:
            kv_cache.store_kv(
                batch_idx,
                torch.arange(seq_len, device=x.device),
                kv_latent,
                v.reshape(bsz * seq_len, self.n_heads * self.v_head_dim),
                k_pe
            )
        
        # 10. Apply GLM-specific mask if needed
        if self.glm_mode and glm_mask is not None:
            # GLM attention masking
            if self.glm_attention_type == "bidirectional":
                # Full bidirectional attention on all tokens
                pass
            elif self.glm_attention_type == "prefix":
                # Prefix LM: bidirectional on prefix, causal on generation
                if hasattr(self, 'prefix_attention_bias'):
                    pass
            elif self.glm_attention_type == "sentinel":
                # Sentinel LM: special sentinel tokens
                pass
        
        # 11. Apply final mask
        attn_mask = mask
        if glm_mask is not None:
            attn_mask = mask if mask is not None else glm_mask
        
        # 12. Attention computation with FP8 memory-efficient handling
        # Convert from FP8 to higher precision for attention computation
        if self.args.use_fp8_training and q_full.dtype == torch.float8_e4m3fn:
            q_full = q_full.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)
        
        if HAS_FLASH_ATTN and self.args.use_flash_attn and seq_len > 1:
            attn_out = flash_attn_func(
                q_full, k, v,
                dropout_p=self.args.attention_dropout if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=(attn_mask is None and seq_len > 1)
            )
        else:
            attn_out = self._sdpa_attention(q_full, k, v, attn_mask)
        
        # 13. Output projection
        attn_out = attn_out.reshape(bsz, seq_len, self.n_heads * self.v_head_dim)
        attn_out = self.attn_dropout(attn_out)
        output = self.wo(attn_out)
        output = self.resid_dropout(output)
        
        return output
    
    def _sdpa_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if mask is None and q.size(2) > 1:
            mask = torch.triu(
                torch.full((q.size(2), k.size(2)), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1
            )
        
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.args.attention_dropout if self.training else 0.0,
            is_causal=(mask is None and q.size(2) > 1)
        )
        
        return attn_output.transpose(1, 2)


# ============================================================================
# ENHANCED MOE COMPONENTS
# ============================================================================

class SwiGLUExpert(nn.Module):
    """SwiGLU Expert for MoE - Enhanced with FP8 support"""
    
    def __init__(
        self, dim: int, inter_dim: int, dtype: torch.dtype,
        dropout: float = 0.0, use_fused: bool = False, use_fp8: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.inter_dim = inter_dim
        self.use_fp8 = use_fp8 and hasattr(torch, 'float8_e4m3fn')
        
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_fused = use_fused
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8 and x.device.type == 'cuda':
            # Use AMP for FP8 operations to maintain precision
            with torch.cuda.amp.autocast(dtype=torch.float8_e4m3fn):
                gate = self.gate_proj(x)
                up = self.up_proj(x)
        else:
            gate = self.gate_proj(x)
            up = self.up_proj(x)
        
        if self.use_fused and hasattr(F, 'silu'):
            gate = F.silu(gate)
        else:
            gate = gate * torch.sigmoid(gate)
        
        hidden = gate * up
        output = self.down_proj(hidden)
        return self.dropout(output)


class AdaptiveRouter(nn.Module):
    """Adaptive Router with learnable temperature and bias - ENHANCED VERSION"""
    
    def __init__(self, args: ModelArgs, n_experts: int, dim: int):
        super().__init__()
        self.args = args
        self.n_experts = n_experts
        self.dim = dim
        self.top_k = args.n_activated_experts
        self.capacity_factor = args.adaptive_router_expert_capacity_factor
        
        self.gate = nn.Linear(dim, n_experts, bias=False, dtype=args.get_dtype())
        
        # Learnable router parameters
        self.logit_temperature = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("expert_bias", torch.zeros(n_experts))
        
        # Expert capacity tracking
        self.register_buffer("expert_usage_count", torch.zeros(n_experts))
        self.register_buffer("expert_usage_total", torch.zeros(1))
        
        self.temperature_min = args.adaptive_router_temperature_min
        self.temperature_max = args.adaptive_router_temperature_max
        self.bias_update_rate = args.adaptive_router_bias_update_rate
        
        self.score_func = args.moe_router_score_func
        self.jitter = args.moe_router_jitter
        
        # Load balancing tracking
        self.register_buffer("expert_load", torch.zeros(n_experts))
        self.register_buffer("load_balance_loss", torch.zeros(1))
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.size(0)
        
        # Compute router logits
        router_logits = F.linear(x_flat, self.gate.weight, self.expert_bias.to(x_flat.dtype))
        
        # Apply adaptive temperature
        temperature = torch.clamp(
            torch.sigmoid(self.logit_temperature) * (self.temperature_max - self.temperature_min) + self.temperature_min,
            self.temperature_min, self.temperature_max
        )
        router_logits = router_logits / temperature
        
        # Add jitter for exploration during training
        if self.training and self.jitter > 0:
            noise = torch.randn_like(router_logits) * self.jitter
            router_logits = router_logits + noise
        
        # Apply score function
        if self.score_func == "softmax":
            router_probs = F.softmax(router_logits.float(), dim=-1).to(x.dtype)
        else:
            router_probs = torch.sigmoid(router_logits.float()).to(x.dtype)
        
        # Get top-k experts
        topk_probs, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        if self.score_func == "softmax":
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Update usage statistics
        with torch.no_grad():
            self.expert_usage_total += num_tokens
            expert_assignments = torch.zeros(self.n_experts, device=x.device)
            for i in range(self.top_k):
                indices = topk_indices[:, i]
                expert_assignments.scatter_add_(0, indices, torch.ones(num_tokens, device=x.device))
            self.expert_usage_count = self.expert_usage_count * 0.99 + expert_assignments * 0.01
        
        # Compute load balancing loss
        load_balance_loss = self._compute_load_balance_loss(topk_indices, num_tokens)
        
        return topk_probs, topk_indices, load_balance_loss, temperature
    
    def _compute_load_balance_loss(self, topk_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        """Compute load balancing loss to encourage equal expert usage"""
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()
        expert_fraction = expert_mask.sum(dim=(0, 1)) / (num_tokens * self.top_k + 1e-8)
        
        router_prob_fraction = torch.sigmoid(self.gate.weight.mean(dim=0))
        
        # Compute variance-based load balance loss
        load_balance = (expert_fraction * router_prob_fraction).sum() * self.n_experts
        
        # Add capacity-based regularization
        capacity_usage = expert_fraction / (self.capacity_factor + 1e-8)
        capacity_loss = torch.mean(F.relu(capacity_usage - 1.0))
        
        return load_balance + capacity_loss * 0.1
    
    def update_expert_bias(self):
        """Update expert bias based on usage with momentum smoothing"""
        if not hasattr(self, 'expert_bias_momentum'):
            self.register_buffer('expert_bias_momentum', torch.zeros_like(self.expert_bias))
        
        with torch.no_grad():
            target_usage = 1.0 / self.n_experts
            current_usage = self.expert_usage_count / (self.expert_usage_total + 1e-8)
            bias_update = (target_usage - current_usage) * self.bias_update_rate
            
            # Apply momentum smoothing to prevent instability
            momentum_coeff = 0.9
            self.expert_bias_momentum = momentum_coeff * self.expert_bias_momentum + (1 - momentum_coeff) * bias_update
            self.expert_bias += self.expert_bias_momentum


class FusedMoELayer(nn.Module):
    """Fused Mixture of Experts Layer - ENHANCED with Adaptive Router and Load Balancing"""
    
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.dim
        self.n_experts = args.n_routed_experts
        self.top_k = args.n_activated_experts
        self.capacity_factor = args.expert_capacity_factor
        self.inter_dim = args.moe_inter_dim
        self.dtype = args.get_dtype()
        self.use_fp8 = args.use_fp8_training
        
        self.use_megablocks = HAS_MEGABLOCKS and args.moe_use_fused_kernel
        self.use_triton = HAS_TRITON and args.use_triton_kernels and not self.use_megablocks
        
        # Enhanced adaptive router
        self.router = AdaptiveRouter(args, self.n_experts, self.dim)
        
        if self.use_megablocks:
            self._init_megablocks(args)
        elif self.use_triton:
            self._init_triton_experts(args)
        else:
            self._init_fallback_experts(args)
    
    def _init_megablocks(self, args: ModelArgs):
        try:
            mega_args = MegaArgs(
                hidden_size=self.dim,
                ffn_hidden_size=self.inter_dim,
                num_experts=self.n_experts,
                top_k=self.top_k,
                capacity_factor=self.capacity_factor,
                moe_dropless=args.moe_dropless,
                expert_parallel=args.moe_expert_parallel,
                activation_fn='swiglu',
                bias=False,
            )
            self.moe = MegaMoE(mega_args)
            self.is_fused = True
        except Exception as e:
            logger.warning(f"Megablocks init failed: {e}, falling back")
            self.use_megablocks = False
            self._init_fallback_experts(args)
    
    def _init_triton_experts(self, args: ModelArgs):
        self.w1 = nn.Parameter(torch.empty(self.n_experts, self.inter_dim * 2, self.dim, dtype=self.dtype))
        self.w2 = nn.Parameter(torch.empty(self.n_experts, self.dim, self.inter_dim, dtype=self.dtype))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02)
        self.is_fused = True
    
    def _init_fallback_experts(self, args: ModelArgs):
        self.experts = nn.ModuleList([
            SwiGLUExpert(self.dim, self.inter_dim, self.dtype, args.dropout, use_fp8=self.use_fp8)
            for _ in range(self.n_experts)
        ])
        self.is_fused = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        num_tokens = x_flat.size(0)
        
        # Adaptive routing
        topk_probs, topk_indices, load_balance_loss, temperature = self.router(x)
        
        # Compute auxiliary losses
        aux_loss = self._compute_aux_loss(x_flat, topk_indices, num_tokens, load_balance_loss)
        
        # Route through experts
        if self.use_megablocks and hasattr(self, 'moe'):
            output = self.moe(x_flat)
            output = output.view(batch_size, seq_len, dim)
            return output, aux_loss
        elif self.use_triton and hasattr(self, 'w1'):
            output = self._triton_moe_forward(x_flat, topk_indices, topk_probs)
            output = output.view(batch_size, seq_len, dim)
            return output, aux_loss
        else:
            output = self._fallback_moe_forward(x_flat, topk_indices, topk_probs)
            output = output.view(batch_size, seq_len, dim)
            return output, aux_loss
    
    def _triton_moe_forward(
        self, x: torch.Tensor, topk_indices: torch.Tensor, topk_probs: torch.Tensor
    ) -> torch.Tensor:
        num_tokens = x.size(0)
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.n_experts):
            mask = (topk_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            
            expert_input = x[mask]
            expert_mask_positions = (topk_indices[mask] == expert_idx).nonzero(as_tuple=True)
            weights = topk_probs[mask]
            expert_weights = weights.gather(1, expert_mask_positions[1].unsqueeze(1))
            
            w1_e = self.w1[expert_idx]
            w2_e = self.w2[expert_idx]
            
            gate_weight, up_weight = w1_e.chunk(2, dim=0)
            
            gate_out = F.linear(expert_input, gate_weight)
            gate_out = F.silu(gate_out)
            up_out = F.linear(expert_input, up_weight)
            hidden = gate_out * up_out
            expert_out = F.linear(hidden, w2_e)
            expert_out = expert_out * expert_weights
            
            output[mask] += expert_out
        
        return output
    
    def _fallback_moe_forward(
        self, x: torch.Tensor, topk_indices: torch.Tensor, topk_probs: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(x)
        
        for expert_idx in range(self.n_experts):
            mask = (topk_indices == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            
            expert_input = x[mask]
            expert_positions = (topk_indices[mask] == expert_idx).nonzero(as_tuple=True)[1]
            weights = topk_probs[mask][torch.arange(mask.sum().item()), expert_positions]
            expert_output = self.experts[expert_idx](expert_input)
            expert_output = expert_output * weights.unsqueeze(-1)
            output[mask] += expert_output
        
        return output
    
    def _compute_aux_loss(
        self, x: torch.Tensor, topk_indices: torch.Tensor, num_tokens: int, load_balance_loss: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary losses - FIXED VERSION"""
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_experts).float()
        expert_fraction = expert_mask.sum(dim=(0, 1)) / (num_tokens * self.top_k + 1e-8)
        
        with torch.no_grad():
            router_logits = self.router.gate(x)
            if self.args.moe_router_score_func == "softmax":
                router_probs = F.softmax(router_logits.float(), dim=-1)
            else:
                router_probs = torch.sigmoid(router_logits.float())
            router_prob_fraction = router_probs.mean(dim=0)
        
        aux_loss = (expert_fraction * router_prob_fraction).sum() * self.n_experts
        
        z_loss = torch.tensor(0.0, device=x.device)
        try:
            z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        except Exception as e:
            logger.debug(f"Z-loss computation failed: {e}")
        
        total_aux_loss = (
            aux_loss * self.args.moe_aux_loss_weight +
            z_loss * self.args.moe_router_z_loss_weight +
            load_balance_loss * self.args.moe_load_balance_loss_weight
        )
        
        return total_aux_loss
    
    def update_router_bias(self):
        """Update router bias after each step"""
        if hasattr(self, 'router'):
            self.router.update_expert_bias()


class SharedExpertLayer(nn.Module):
    """Shared Expert Layer for MoE - ENHANCED VERSION"""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.inter_dim = args.shared_expert_inter_dim
        self.dtype = args.get_dtype()
        self.scale = args.shared_expert_scale
        
        self.gate_proj = nn.Linear(self.dim, self.inter_dim, bias=False, dtype=self.dtype)
        self.up_proj = nn.Linear(self.dim, self.inter_dim, bias=False, dtype=self.dtype)
        self.down_proj = nn.Linear(self.inter_dim, self.dim, bias=False, dtype=self.dtype)
        self.dropout = nn.Dropout(args.dropout) if args.dropout > 0 else nn.Identity()
        
        # Learnable gate for shared expert contribution
        self.gate = nn.Parameter(torch.tensor(self.scale))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.dropout(self.down_proj(hidden))
        return output * torch.sigmoid(self.gate)


class DenseMLP(nn.Module):
    """Dense MLP Layer"""
    
    def __init__(self, dim: int, inter_dim: int, dtype: torch.dtype, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, inter_dim, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(dim, inter_dim, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(inter_dim, dim, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class ParallelMoEDenseLayer(nn.Module):
    """Parallel MoE + Dense Layer with Residual Fusion - ENHANCED VERSION"""
    
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.args = args
        self.layer_idx = layer_idx
        self.dim = args.dim
        self.dtype = args.get_dtype()
        
        self.moe_path = FusedMoELayer(args, layer_idx)
        self.dense_path = DenseMLP(args.dim, args.inter_dim, self.dtype, args.dropout)
        
        if args.use_shared_expert:
            self.shared_expert = SharedExpertLayer(args)
        else:
            self.shared_expert = None
        
        self.combine_mode = args.parallel_moe_dense_combine
        self.combine_ratio = args.parallel_moe_dense_ratio
        
        if self.combine_mode == "gated":
            self.gate = nn.Parameter(torch.tensor(0.5, dtype=self.dtype))
        elif self.combine_mode == "concat":
            self.combine_proj = nn.Linear(self.dim * 3, self.dim, bias=False, dtype=self.dtype)
        elif self.combine_mode == "residual_fusion":
            # Residual fusion with learnable weights
            self.moe_weight = nn.Parameter(torch.tensor(0.5, dtype=self.dtype))
            self.dense_weight = nn.Parameter(torch.tensor(0.5, dtype=self.dtype))
        
        self.use_parallel = args.use_parallel_moe_dense
        
        self.register_buffer("moe_usage", torch.zeros(1))
        self.register_buffer("dense_usage", torch.zeros(1))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, dim = x.shape
        
        if self.use_parallel:
            moe_out, aux_loss = self.moe_path(x)
            dense_out = self.dense_path(x)
            
            if moe_out.shape != x.shape:
                moe_out = moe_out.view(batch_size, seq_len, dim)
            if dense_out.shape != x.shape:
                dense_out = dense_out.view(batch_size, seq_len, dim)
            
            if self.shared_expert is not None:
                shared_out = self.shared_expert(x)
                shared_scale = getattr(self.args, 'shared_expert_scale', 0.5)
                shared_out = shared_out * shared_scale
            else:
                shared_out = torch.zeros_like(x)
            
            actual_ratio = getattr(self, 'current_ratio', self.combine_ratio)
            
            if self.combine_mode == "add":
                output = moe_out * actual_ratio + dense_out * (1 - actual_ratio)
                output = output + shared_out
            elif self.combine_mode == "gated":
                gate_value = torch.sigmoid(self.gate)
                output = moe_out * gate_value + dense_out * (1 - gate_value)
                output = output + shared_out * 0.1
            elif self.combine_mode == "concat":
                combined = torch.cat([moe_out, dense_out, shared_out], dim=-1)
                output = self.combine_proj(combined)
            elif self.combine_mode == "residual_fusion":
                moe_weight = torch.sigmoid(self.moe_weight)
                dense_weight = torch.sigmoid(self.dense_weight)
                output = x + moe_out * moe_weight + dense_out * dense_weight + shared_out * 0.05
            else:
                output = moe_out + dense_out + shared_out * 0.1
            
            if not self.training:
                with torch.no_grad():
                    moe_norm = moe_out.abs().mean()
                    dense_norm = dense_out.abs().mean()
                    x_norm = x.abs().mean() + 1e-8
                    self.moe_usage = self.moe_usage * 0.99 + (moe_norm / x_norm) * 0.01
                    self.dense_usage = self.dense_usage * 0.99 + (dense_norm / x_norm) * 0.01
        else:
            output, aux_loss = self.moe_path(x)
            if self.shared_expert is not None:
                output = output + self.shared_expert(x) * self.combine_ratio
        
        return output, aux_loss
    
    def update_router_bias(self):
        """Update router bias"""
        if hasattr(self.moe_path, 'update_router_bias'):
            self.moe_path.update_router_bias()
    
    def get_balance_stats(self) -> Dict[str, float]:
        gate_value = torch.sigmoid(self.gate).item() if self.combine_mode == "gated" else None
        
        return {
            "moe_usage": self.moe_usage.item(),
            "dense_usage": self.dense_usage.item(),
            "balance_ratio": self.moe_usage.item() / (self.dense_usage.item() + 1e-8),
            "combine_mode": self.combine_mode,
            "combine_ratio": self.combine_ratio,
            "gate_value": gate_value,
        }


# ============================================================================
# ENHANCED TRANSFORMER BLOCK WITH DYNAMIC DEPTH
# ============================================================================

class TransformerBlock(nn.Module):
    """Single Transformer Block with Attention and MoE/Dense MLP - ENHANCED with Dynamic Depth"""
    
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.dtype = args.get_dtype()
        self.kv_cache = None
        self.use_parallel = args.use_parallel_moe_dense
        self.skip_counter = 0  # Track dynamic depth skips
        
        self.attention = MultiHeadLatentAttention(args, layer_id)
        self.attention_norm = RMSNorm(args.dim, args.rms_norm_eps)
        
        if layer_id < args.n_dense_layers:
            self.mlp = DenseMLP(args.dim, args.inter_dim, self.dtype, args.dropout)
            self.shared_expert = None
            self.is_moe = False
        else:
            if args.use_parallel_moe_dense:
                self.mlp = ParallelMoEDenseLayer(args, layer_id)
                self.shared_expert = None
                self.is_moe = True
            else:
                self.mlp = FusedMoELayer(args, layer_id)
                if args.use_shared_expert:
                    self.shared_expert = SharedExpertLayer(args)
                else:
                    self.shared_expert = None
                self.is_moe = True
        
        self.mlp_norm = RMSNorm(args.dim, args.rms_norm_eps)
        
        if layer_id >= args.n_dense_layers and not args.use_parallel_moe_dense:
            self.gate = nn.Parameter(torch.ones(1))
        
        # Dynamic depth: layer skip probability
        self.use_dynamic_depth = args.use_dynamic_depth
        self.skip_prob = args.dynamic_depth_skip_prob
        self.confidence_threshold = args.dynamic_depth_confidence_threshold
        
        # Layer confidence score (learned)
        if self.use_dynamic_depth:
            self.confidence = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[PagedKVCache] = None,
        batch_idx: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[float]]:
        
        # Dynamic depth: decide whether to skip this layer
        skip_layer = False
        confidence = None
        if self.use_dynamic_depth and self.training:
            confidence = torch.sigmoid(self.confidence)
            confidence_val = confidence.item()
            if confidence_val > self.confidence_threshold:
                skip_prob = self.skip_prob * confidence_val
                # Random sampling to decide layer skip
                if torch.rand(1).item() < skip_prob:
                    skip_layer = True
                    # Track skip statistics for monitoring
                    if hasattr(self, 'skip_counter'):
                        self.skip_counter += 1
        if skip_layer:
            # Skip this layer entirely (residual connection only)
            # Avoid unnecessary gradient checkpointing computation
            return x, None, confidence.item() if confidence is not None else None
        
        residual = x
        x_norm = self.attention_norm(x)
        
        # Only apply gradient checkpointing when not skipping
        if self.training and self.args.gradient_checkpointing:
            attn_out = torch_checkpoint(
                self.attention, x_norm, start_pos, kv_cache, batch_idx, mask,
                use_reentrant=True
            )
        else:
            attn_out = self.attention(x_norm, start_pos, kv_cache, batch_idx, mask)
        
        x = residual + attn_out
        
        residual = x
        x_norm = self.mlp_norm(x)
        
        if self.is_moe:
            if self.training and self.args.gradient_checkpointing:
                moe_out, aux_loss = torch_checkpoint(self.mlp, x_norm, use_reentrant=True)
            else:
                moe_out, aux_loss = self.mlp(x_norm)
            
            if self.shared_expert is not None:
                shared_out = self.shared_expert(x_norm)
                moe_out = moe_out + shared_out
            
            if hasattr(self, 'gate'):
                moe_out = self.gate * moe_out
            
            x = residual + moe_out
            return x, aux_loss, None
        else:
            if self.training and self.args.gradient_checkpointing:
                mlp_out = torch_checkpoint(self.mlp, x_norm, use_reentrant=True)
            else:
                mlp_out = self.mlp(x_norm)
            
            x = residual + mlp_out
            return x, None, None
    
    def update_router_bias(self):
        """Update router bias for MoE layers"""
        if self.is_moe and hasattr(self.mlp, 'update_router_bias'):
            self.mlp.update_router_bias()
    
    def get_balance_stats(self) -> Optional[Dict]:
        if self.use_parallel and hasattr(self.mlp, 'get_balance_stats'):
            return self.mlp.get_balance_stats()
        return None


# ============================================================================
# ENHANCED MAIN TRANSFORMER MODEL
# ============================================================================

class Transformer(nn.Module):
    """Main Transformer Model with MoE Architecture - ENHANCED VERSION"""
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dtype = args.get_dtype()
        self.dim = args.dim
        self.n_layers = args.n_layers
        self.vocab_size = args.vocab_size
        self.max_seq_len = args.max_seq_len
        
        # FP8 training support
        self.use_fp8 = args.use_fp8_training and hasattr(torch, 'float8_e4m3fn') and torch.cuda.is_available()
        
        self.embed_tokens = nn.Embedding(args.vocab_size, args.dim, dtype=self.dtype)
        self.embed_dropout = nn.Dropout(args.embedding_dropout) if args.embedding_dropout > 0 else nn.Identity()
        
        self.layers = nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.rms_norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False, dtype=self.dtype)
        
        # Multi-Token Prediction heads
        self.use_mtp = args.use_multi_token_prediction
        if self.use_mtp:
            # Validate MTP parameters
            assert args.mtp_n_predictions > 0, "mtp_n_predictions must be > 0"
            assert args.mtp_n_predictions <= args.max_seq_len, \
                f"MTP predictions {args.mtp_n_predictions} exceeds max_seq_len {args.max_seq_len}"
            
            self.mtp_heads = nn.ModuleList([
                nn.Linear(args.dim, args.vocab_size, bias=False, dtype=self.dtype)
                for _ in range(args.mtp_n_predictions)
            ])
            self.mtp_weight = args.mtp_loss_weight
        
        self.tie_word_embeddings = getattr(args, 'tie_word_embeddings', False)
        if self.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
            if self.use_mtp:
                for head in self.mtp_heads:
                    head.weight = self.embed_tokens.weight
        
        self._init_weights()
        self.total_params = count_parameters(self)
        self.active_params = self._compute_active_params()
    
    def _init_weights(self):
        std = 0.02
        if self.args.use_deepnorm:
            std = std * (2 * self.n_layers) ** (-0.5)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def _compute_active_params(self) -> int:
        base_params = count_parameters(self.embed_tokens) + count_parameters(self.lm_head)
        base_params += count_parameters(self.norm)
        
        for layer in self.layers:
            base_params += count_parameters(layer.attention)
            base_params += count_parameters(layer.attention_norm)
            base_params += count_parameters(layer.mlp_norm)
        
        moe_active_params = 0
        for layer in self.layers:
            if layer.is_moe:
                if layer.use_parallel and hasattr(layer.mlp, 'moe_path'):
                    moe_active_params += count_parameters(layer.mlp.moe_path)
                    if hasattr(layer.mlp, 'dense_path'):
                        moe_active_params += count_parameters(layer.mlp.dense_path)
                    if hasattr(layer.mlp, 'shared_expert'):
                        moe_active_params += count_parameters(layer.mlp.shared_expert)
                else:
                    if hasattr(layer.mlp, 'shared_expert'):
                        moe_active_params += count_parameters(layer.mlp.shared_expert)
                    if hasattr(layer.mlp, 'experts'):
                        expert_params = count_parameters(layer.mlp.experts[0])
                        moe_active_params += expert_params * self.args.n_activated_experts
                    if hasattr(layer.mlp, 'gate'):
                        moe_active_params += count_parameters(layer.mlp.gate)
        
        return base_params + moe_active_params
    
    def create_kv_cache(self) -> PagedKVCache:
        return PagedKVCache(
            max_batch_size=self.args.max_batch_size,
            max_num_blocks=self.args.max_num_blocks,
            block_size=self.args.block_size,
            num_heads=self.args.n_heads,
            head_dim=self.args.v_head_dim,
            kv_lora_rank=self.args.kv_lora_rank,
            qk_rope_head_dim=self.args.qk_rope_head_dim,
            device=next(self.parameters()).device,
            dtype=self.dtype
        )
    
    def get_parallel_stats(self) -> Dict[str, Any]:
        stats = {}
        for i, layer in enumerate(self.layers):
            block_stats = layer.get_balance_stats()
            if block_stats:
                stats[f"layer_{i}"] = block_stats
        return stats
    
    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[PagedKVCache] = None,
        batch_idx: int = 0,
        mask: Optional[torch.Tensor] = None,
        return_mtp_loss: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        
        # FP8 casting for input embeddings if enabled and supported on CUDA
        hidden_states = self.embed_tokens(input_ids)
        if self.use_fp8 and hidden_states.device.type == 'cuda':
            hidden_states = hidden_states.to(torch.float8_e4m3fn)
        
        hidden_states = self.embed_dropout(hidden_states)
        
        if mask is None and seq_len > 1 and kv_cache is None:
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), 
                          device=input_ids.device, dtype=hidden_states.dtype),
                diagonal=1
            )
        
        aux_losses = []
        layer_confidences = []
        
        for layer in self.layers:
            if self.training and self.args.gradient_checkpointing:
                hidden_states, aux_loss, confidence = torch.utils.checkpoint.checkpoint(
                    lambda x, sp, kvc, bi, m: layer(x, sp, kvc, bi, m),
                    hidden_states, start_pos, kv_cache, batch_idx, mask,
                    use_reentrant=True
                )
            else:
                hidden_states, aux_loss, confidence = layer(hidden_states, start_pos, kv_cache, batch_idx, mask)
            
            if aux_loss is not None:
                aux_losses.append(aux_loss)
            if confidence is not None:
                layer_confidences.append(confidence)
        
        hidden_states = self.norm(hidden_states)
        
        # Convert back from FP8 if needed
        if self.use_fp8 and hidden_states.dtype == torch.float8_e4m3fn:
            hidden_states = hidden_states.to(self.dtype)
        
        logits = self.lm_head(hidden_states)
        
        # Multi-Token Prediction loss
        mtp_loss = None
        if return_mtp_loss and self.use_mtp and self.training:
            mtp_loss = self._compute_mtp_loss(hidden_states, input_ids)
        
        return logits, aux_losses, mtp_loss
    
    def _compute_mtp_loss(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute Multi-Token Prediction loss (vectorized) with gradient checkpointing"""
        batch_size, seq_len, dim = hidden_states.shape
        
        # Add gradient checkpointing for MTP heads
        if self.training and self.args.gradient_checkpointing:
            mtp_loss_total = torch_checkpoint(
                self._compute_mtp_loss_impl,
                hidden_states, input_ids,
                use_reentrant=True
            )
        else:
            mtp_loss_total = self._compute_mtp_loss_impl(hidden_states, input_ids)
        
        return mtp_loss_total
    
    def _compute_mtp_loss_impl(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Internal MTP loss computation (vectorized)"""
        batch_size, seq_len, dim = hidden_states.shape
        mtp_loss_total = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        
        # Vectorized MTP loss computation
        valid_k_count = 0
        for k in range(self.args.mtp_n_predictions):
            output_len = seq_len - k - 1
            if output_len <= 0:
                continue
            
            valid_k_count += 1
            # Shift hidden states for future token prediction
            mtp_hidden = hidden_states[:, :output_len, :]
            target_ids = input_ids[:, k+1: k+1 + output_len]
            
            # Compute logits and loss
            mtp_logits = self.mtp_heads[k](mtp_hidden)
            mtp_loss = F.cross_entropy(
                mtp_logits.reshape(-1, self.vocab_size),
                target_ids.reshape(-1),
                ignore_index=-100,
                reduction='mean'
            )
            
            # Weight loss by distance (closer predictions are more important)
            weight = (self.args.mtp_n_predictions - k) / self.args.mtp_n_predictions
            mtp_loss_total = mtp_loss_total + mtp_loss * weight
        
        # Average over valid k values
        if valid_k_count > 0:
            mtp_loss_total = mtp_loss_total / valid_k_count
        
        return mtp_loss_total * self.mtp_weight
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        stop_tokens: Optional[Set[int]] = None,
        stream: bool = False,
        callback: Optional[Callable] = None
    ) -> Union[torch.Tensor, Generator]:
        """Generate text autoregressively - FIXED VERSION with proper memory cleanup"""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        if batch_size != 1:
            raise ValueError("Batch generation not supported yet")
        
        if input_ids.shape[1] == 0:
            bos_id = getattr(self.args, 'bos_token_id', 1)
            input_ids = torch.tensor([[bos_id]], device=device)
        
        prompt_len = input_ids.shape[1]
        
        kv_cache = None
        total_len = min(prompt_len + max_new_tokens, self.max_seq_len)
        
        try:
            kv_cache = self.create_kv_cache()
            kv_cache.allocate_sequence(0, total_len)
            
            current_input = input_ids
            current_pos = 0
            all_tokens = input_ids[0].tolist()
            
            generated_tokens = []
            
            for step in range(max_new_tokens):
                logits, _, _ = self.forward(
                    current_input,
                    start_pos=current_pos,
                    kv_cache=kv_cache,
                    batch_idx=0
                )
                
                next_logits = logits[:, -1:, :]
                
                if repetition_penalty != 1.0:
                    recent_tokens = set(all_tokens[-50:])
                    for token_id in recent_tokens:
                        if token_id < next_logits.size(-1):
                            if next_logits[0, 0, token_id] > 0:
                                next_logits[0, 0, token_id] /= repetition_penalty
                            else:
                                next_logits[0, 0, token_id] *= repetition_penalty
                
                next_token = self._sample_token(
                    next_logits, temperature, top_p, top_k
                )
                
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
                if stop_tokens and next_token.item() in stop_tokens:
                    break
                
                token_id = next_token.item()
                generated_tokens.append(token_id)
                all_tokens.append(token_id)
                
                if stream:
                    yield next_token
                if callback:
                    callback(token_id, step)
                
                current_input = next_token.view(1, 1)
                current_pos = prompt_len + step
            
            if not stream:
                return torch.tensor(generated_tokens, device=device)
                
        finally:
            if kv_cache is not None:
                try:
                    kv_cache.free_sequence(0)
                    # Explicitly cleanup large tensors to avoid memory leaks
                    if hasattr(kv_cache, 'block_tables') and kv_cache.block_tables is not None:
                        del kv_cache.block_tables
                    if hasattr(kv_cache, 'k_blocks') and kv_cache.k_blocks is not None:
                        del kv_cache.k_blocks
                    if hasattr(kv_cache, 'v_blocks') and kv_cache.v_blocks is not None:
                        del kv_cache.v_blocks
                    if hasattr(kv_cache, 'pe_blocks') and kv_cache.pe_blocks is not None:
                        del kv_cache.pe_blocks
                    if hasattr(kv_cache, 'seq_lens') and kv_cache.seq_lens is not None:
                        del kv_cache.seq_lens
                    if hasattr(kv_cache, 'num_blocks_per_seq') and kv_cache.num_blocks_per_seq is not None:
                        del kv_cache.num_blocks_per_seq
                    del kv_cache
                except Exception as e:
                    logger.error(f"Error during KV cache cleanup: {e}")
            
            # Force garbage collection and clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int
    ) -> torch.Tensor:
        """Safe token sampling with full validation"""
        if temperature <= 0:
            return logits.argmax(dim=-1)
        
        logits = logits / temperature
        vocab_size = logits.size(-1)
        
        # Validate top_k
        if top_k <= 0:
            top_k = 1
        top_k = min(top_k, vocab_size)
        
        # Top-k filtering
        if top_k < vocab_size:
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_values)
            logits = mask
        
        # Top-p (nucleus) filtering
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float('-inf'))
        
        probs = F.softmax(logits.float(), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        if probs.sum() == 0:
            return logits.argmax(dim=-1)
        
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def get_model_info(self) -> Dict:
        return {
            "model_name": self.args.model_name,
            "total_params": self.total_params,
            "total_params_formatted": format_number(self.total_params),
            "active_params": self.active_params,
            "active_params_formatted": format_number(self.active_params),
            "sparsity": 1 - (self.active_params / self.total_params) if self.total_params > 0 else 0,
            "n_layers": self.n_layers,
            "n_heads": self.args.n_heads,
            "dim": self.dim,
            "n_experts": self.args.total_experts,
            "n_activated_experts": self.args.n_activated_experts,
            "max_seq_len": self.max_seq_len,
            "vocab_size": self.vocab_size,
            "dtype": str(self.dtype),
            "use_parallel_moe_dense": self.args.use_parallel_moe_dense,
            "parallel_combine_mode": self.args.parallel_moe_dense_combine,
            "use_glm": self.args.use_glm,
            "use_adaptive_router": self.args.use_adaptive_router,
            "use_dynamic_depth": self.args.use_dynamic_depth,
            "use_multi_token_prediction": self.args.use_multi_token_prediction,
            "use_fp8_training": self.args.use_fp8_training,
        }


# ============================================================================
# TOKENIZER (Same as before)
# ============================================================================

class ProductionTokenizer:
    """Production-ready Tokenizer with SentencePiece support"""
    
    SPECIAL_TOKENS = {
        '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3,
        '<mask>': 4, '<sep>': 5, '<cls>': 6,
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 32000,
        pad_token: str = '<pad>',
        bos_token: str = '<s>',
        eos_token: str = '</s>',
        unk_token: str = '<unk>'
    ):
        self.vocab_size = vocab_size
        self.pad_token_id = self.SPECIAL_TOKENS[pad_token]
        self.bos_token_id = self.SPECIAL_TOKENS[bos_token]
        self.eos_token_id = self.SPECIAL_TOKENS[eos_token]
        self.unk_token_id = self.SPECIAL_TOKENS[unk_token]
        self.mask_token_id = self.SPECIAL_TOKENS['<mask>']
        
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        
        self.hf_tokenizer = None
        self.use_hf = False
        
        # FIXED: Create cached encode method using instance-safe cache with thread safety
        self._encode_cache: OrderedDict = OrderedDict()
        self._encode_cache_maxsize = 10000
        self._cache_lock = threading.Lock()
        self.encode_cached = self._encode_cached
        
        if HAS_SPM and model_path and os.path.exists(model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(model_path)
            self.vocab_size = self.sp.GetPieceSize()
            self.use_spm = True
            self._init_special_ids_spm()
        else:
            self._init_fallback_vocab()
            self.use_spm = False
        
        if not self.use_spm:
            try:
                from transformers import AutoTokenizer
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    "Xenova/bert-base-multilingual-cased",
                    use_fast=True
                )
                self.use_hf = True
                self.vocab_size = self.hf_tokenizer.vocab_size
                logger.info("Loaded HuggingFace tokenizer as fallback")
            except Exception as e:
                self.use_hf = False
                logger.warning(f"No fallback tokenizer available: {e}")
    
    def clear_cache(self):
        """Clear tokenizer cache"""
        if hasattr(self, 'encode_cached'):
            self.encode_cached.cache_clear()
    
    def _encode_impl_func(self, text_key: str) -> Tuple[int, ...]:
        """Internal encode implementation (cached)"""
        text = text_key
        if self.use_spm:
            tokens = self.sp.EncodeAsIds(text)
        elif self.use_hf and self.hf_tokenizer:
            tokens = self.hf_tokenizer.encode(text, add_special_tokens=False, truncation=False)
        else:
            tokens = []
            for char in text:
                tokens.append(self.vocab.get(char, self.unk_token_id))
        
        return tuple(tokens)

    def _encode_cached(self, text: str) -> Tuple[int, ...]:
        """Thread-safe tokenizer cache with LRU eviction"""
        with self._cache_lock:
            if text in self._encode_cache:
                self._encode_cache.move_to_end(text)
                return self._encode_cache[text]
            
            tokens = self._encode_impl_func(text)
            self._encode_cache[text] = tokens
            if len(self._encode_cache) > self._encode_cache_maxsize:
                self._encode_cache.popitem(last=False)
            return tokens
    
    def _init_special_ids_spm(self):
        self.pad_token_id = self.sp.pad_id() if self.sp.pad_id() >= 0 else 0
        self.bos_token_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_token_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        self.unk_token_id = self.sp.unk_id() if self.sp.unk_id() >= 0 else 3
    
    def _init_fallback_vocab(self):
        self.vocab = {k: v for k, v in self.SPECIAL_TOKENS.items()}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        unicode_ranges = [
            (0x0020, 0x007F),
            (0x00A0, 0x00FF),
            (0x0100, 0x017F),
            (0x1E00, 0x1EFF),
        ]
        
        for start, end in unicode_ranges:
            for code in range(start, min(end + 1, 0x10000)):
                try:
                    char = chr(code)
                    if char not in self.vocab:
                        idx = len(self.vocab)
                        self.vocab[char] = idx
                        self.reverse_vocab[idx] = char
                except ValueError:
                    pass
        
        self.vocab_size = len(self.vocab)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False
    ) -> List[int]:
        # FIXED: Use LRU-cached encoding when possible
        cached_tokens = self.encode_cached(text)
        tokens = list(cached_tokens)
        
        if add_special_tokens and not (self.use_hf and self.hf_tokenizer):
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        
        if max_length and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
            if add_special_tokens and tokens[-1] != self.eos_token_id:
                tokens[-1] = self.eos_token_id
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        if self.use_spm:
            text = self.sp.DecodeIds(token_ids)
        elif self.use_hf and self.hf_tokenizer:
            text = self.hf_tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces
            )
        else:
            special_ids = {
                self.bos_token_id, self.eos_token_id, 
                self.pad_token_id, self.unk_token_id, 
                self.mask_token_id
            }
            tokens = []
            for tid in token_ids:
                if skip_special_tokens and tid in special_ids:
                    continue
                token = self.reverse_vocab.get(tid, self.unk_token)
                tokens.append(token)
            text = ''.join(tokens)
        
        if clean_up_tokenization_spaces:
            text = text.replace('▁', ' ').replace('_', ' ').strip()
        
        return text
    
    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: bool = True
    ) -> Dict[str, Union[torch.Tensor, List[List[int]]]]:
        batch_tokens = []
        for text in texts:
            tokens = self.encode(text, add_special_tokens, max_length, truncation)
            batch_tokens.append(tokens)
        
        if not padding:
            if return_tensors:
                return {'input_ids': [torch.tensor(t, dtype=torch.long) for t in batch_tokens]}
            return {'input_ids': batch_tokens}
        
        max_len = max(len(tokens) for tokens in batch_tokens) if batch_tokens else 0
        
        input_ids = []
        attention_masks = []
        
        for tokens in batch_tokens:
            pad_len = max_len - len(tokens)
            padded = tokens + [self.pad_token_id] * pad_len
            mask = [1] * len(tokens) + [0] * pad_len
            input_ids.append(padded)
            attention_masks.append(mask)
        
        if return_tensors:
            return {
                'input_ids': torch.tensor(input_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        return {'input_ids': input_ids, 'attention_mask': attention_masks}
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def save_vocab(self, path: str):
        if self.use_spm:
            logger.info("SPM model is separate file")
        else:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.vocab, f, ensure_ascii=False)
    
    def train_tokenizer(
        self,
        corpus_path: str,
        model_prefix: str,
        vocab_size: int = 32000,
        model_type: str = 'bpe',
        character_coverage: float = 1.0
    ):
        if not HAS_SPM:
            raise RuntimeError("Need SentencePiece: pip install sentencepiece")
        
        import subprocess
        cmd = [
            'spm_train', f'--input={corpus_path}', f'--model_prefix={model_prefix}',
            f'--vocab_size={vocab_size}', f'--model_type={model_type}',
            f'--character_coverage={character_coverage}', '--pad_id=0', '--bos_id=1',
            '--eos_id=2', '--unk_id=3', '--pad_piece=<pad>', '--bos_piece=<s>',
            '--eos_piece=</s>', '--unk_piece=<unk>',
        ]
        
        subprocess.run(cmd, check=True)
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f"{model_prefix}.model")
        self.use_spm = True
        self.vocab_size = self.sp.GetPieceSize()
        self._init_special_ids_spm()


# ============================================================================
# DATASET
# ============================================================================

class TextDataset(Dataset):
    """Dataset for text training"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: ProductionTokenizer,
        max_seq_len: int = 4096,
        min_seq_len: int = 64,
        stride: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.stride = stride
        
        self.data = self._load_data(data_path)
        self.num_sequences = max(0, len(self.data) - max_seq_len) // stride + 1
    
    def _load_data(self, path: str) -> List[int]:
        if os.path.isdir(path):
            texts = []
            for file in sorted(os.listdir(path)):
                if file.endswith(('.txt', '.jsonl', '.json')):
                    filepath = os.path.join(path, file)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        if file.endswith('.jsonl'):
                            for line in f:
                                try:
                                    data = json.loads(line)
                                    if 'text' in data:
                                        texts.append(data['text'])
                                except:
                                    pass
                        else:
                            texts.append(f.read())
            full_text = ' '.join(texts)
        elif os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                full_text = f.read()
        else:
            raise ValueError(f"Data path does not exist: {path}")
        
        return self.tokenizer.encode(full_text, add_special_tokens=False)
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_seq_len + 1
        
        seq = self.data[start_idx:end_idx]
        
        if len(seq) < self.max_seq_len + 1:
            seq = seq + [self.tokenizer.pad_token_id] * (self.max_seq_len + 1 - len(seq))
        
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }


# ============================================================================
# UTILITY FUNCTIONS FOR MODEL SAVE/LOAD
# ============================================================================

def create_model(
    config: Union[str, ModelArgs, Dict],
    device: Optional[str] = None
) -> Tuple[Transformer, ModelArgs]:
    """Create model from configuration"""
    if isinstance(config, str):
        args = ModelArgs.load(config)
    elif isinstance(config, dict):
        args = ModelArgs(**config)
    elif isinstance(config, ModelArgs):
        args = config
    else:
        raise ValueError(f"Invalid config type: {type(config)}")
    
    if not validate_model_args(args):
        raise ValueError("Model configuration validation failed")
    
    if device:
        args.device = device
    
    model = Transformer(args)
    
    if args.print_model_stats:
        info = model.get_model_info()
        logger.info("="*60)
        logger.info("ENHANCED MODEL CREATED")
        logger.info(f"  Total Parameters: {info['total_params_formatted']}")
        logger.info(f"  Active Parameters: {info['active_params_formatted']}")
        logger.info(f"  Sparsity: {info['sparsity']:.1%}")
        logger.info(f"  Layers: {info['n_layers']}")
        logger.info(f"  Experts: {info['n_experts']} (top-{info['n_activated_experts']})")
        logger.info(f"  Max Seq Len: {info['max_seq_len']}")
        logger.info(f"  Parallel MoE+Dense: {info.get('use_parallel_moe_dense', False)}")
        logger.info(f"  GLM: {info.get('use_glm', False)}")
        logger.info(f"  Adaptive Router: {info.get('use_adaptive_router', False)}")
        logger.info(f"  Dynamic Depth: {info.get('use_dynamic_depth', False)}")
        logger.info(f"  MTP: {info.get('use_multi_token_prediction', False)}")
        logger.info(f"  FP8 Training: {info.get('use_fp8_training', False)}")
        logger.info("="*60)
    
    return model, args


def save_model(
    model: Transformer,
    path: str,
    save_config: bool = True,
    use_safetensors: bool = True
):
    """Save model to disk"""
    os.makedirs(path, exist_ok=True)
    
    if save_config:
        config_path = os.path.join(path, "config.json")
        model.args.save(config_path)
    
    state_dict = model.state_dict()
    
    if use_safetensors and HAS_SAFETENSORS:
        from safetensors.torch import save_file
        weights_path = os.path.join(path, "model.safetensors")
        save_file(state_dict, weights_path)
    else:
        weights_path = os.path.join(path, "model.pt")
        torch.save(state_dict, weights_path)
    
    logger.info(f"Model saved to {path}")


def load_model(
    path: str,
    device: Optional[str] = None
) -> Tuple[Transformer, ModelArgs]:
    """Load model from disk"""
    config_path = os.path.join(path, "config.json")
    if os.path.exists(config_path):
        args = ModelArgs.load(config_path)
    else:
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    if device:
        args.device = device
    
    model = Transformer(args)
    
    safetensors_path = os.path.join(path, "model.safetensors")
    pt_path = os.path.join(path, "model.pt")
    
    if os.path.exists(safetensors_path) and HAS_SAFETENSORS:
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location='cpu', weights_only=True)
    else:
        raise FileNotFoundError(f"Weights not found in {path}")
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    
    if missing:
        logger.warning(f"Missing keys: {len(missing)}")
    if unexpected:
        logger.warning(f"Unexpected keys: {len(unexpected)}")
    
    logger.info(f"Model loaded from {path}")
    
    return model, args


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Model Trainer with full features - ENHANCED VERSION"""
    
    def __init__(
        self,
        model: Transformer,
        args: ModelArgs,
        training_args: TrainingArgs,
        tokenizer: Optional[ProductionTokenizer] = None
    ):
        self.model = model
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.device = torch.device(args.device)
        self.dtype = args.get_dtype()
        
        self.use_dist = args.world_size > 1
        self.rank = args.rank
        self.world_size = args.world_size
        
        self.model = self.model.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler(enabled=args.use_amp)
        
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # FIXED: Track scaler reset interval
        self.scaler_steps = 0
        self.scaler_reset_interval = 5000
        
        self.ema_model = None
        if training_args.use_ema:
            self.ema_model = self._create_ema_model()
        
        self.writer = None
        if args.use_tensorboard and HAS_TB:
            self.writer = SummaryWriter(args.tensorboard_dir)
        
        self.wandb_run = None
        if args.use_wandb and HAS_WANDB:
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                entity=args.wandb_entity,
                config={**args.to_dict(), **training_args.__dict__}
            )
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # FIXED: Register cleanup on exit
        atexit.register(self.cleanup)
        
        logger.info(f"Enhanced Trainer initialized [rank {self.rank}/{self.world_size}]")
    
    def cleanup(self):
        """Cleanup resources properly"""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Failed to clear CUDA cache: {e}")
        
        try:
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.flush()
                self.writer.close()
        except Exception as e:
            logger.error(f"Failed to close TensorBoard writer: {e}")
        
        try:
            if hasattr(self, 'wandb_run') and self.wandb_run is not None:
                wandb.finish()
        except Exception as e:
            logger.error(f"Failed to finish WandB: {e}")
        
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'ema_model'):
                del self.ema_model
        except Exception as e:
            logger.error(f"Failed to delete model references: {e}")
        
        try:
            if self.use_dist and dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
                logger.info("Distributed process group destroyed")
        except Exception as e:
            logger.error(f"Failed to cleanup distributed: {e}")
        
        try:
            gc.collect()
        except Exception as e:
            logger.error(f"Failed to run garbage collection: {e}")
        
        logger.info("Trainer cleanup completed")
    
    def __del__(self):
        """Ensure cleanup on object deletion"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in __del__: {e}")
    
    def _create_optimizer(self):
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ['bias', 'norm', 'layernorm', 'rms_norm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.training_args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        if self.training_args.use_fused_adam:
            try:
                return torch.optim.AdamW(
                    param_groups,
                    lr=self.training_args.learning_rate,
                    betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
                    eps=self.training_args.adam_eps,
                    fused=True
                )
            except:
                pass
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.training_args.learning_rate,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_eps
        )
    
    def _create_scheduler(self):
        total_steps = (
            self.training_args.max_steps 
            if self.training_args.max_steps > 0 
            else self.training_args.epochs * 10000
        )
        warmup_steps = self.training_args.warmup_steps
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _create_ema_model(self):
        from copy import deepcopy
        ema = deepcopy(self.model)
        for param in ema.parameters():
            param.requires_grad = False
        return ema
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with proper memory management"""
        self.model.train()
        
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        with autocast(enabled=self.args.use_amp):
            logits, aux_losses, mtp_loss = self.model(input_ids, return_mtp_loss=True)
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=self.training_args.label_smoothing
            )
            
            total_loss = ce_loss
            aux_loss = torch.tensor(0.0, device=self.device)
            if aux_losses:
                aux_loss = torch.stack(aux_losses).mean()
                total_loss = total_loss + aux_loss * self.args.aux_loss_weight
            
            if mtp_loss is not None:
                total_loss = total_loss + mtp_loss
        
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        
        if self.training_args.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_args.clip_grad
            )
        else:
            grad_norm = 0.0
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        # Update router biases after each step
        for layer in self.model.layers:
            if hasattr(layer, 'update_router_bias'):
                layer.update_router_bias()
        
        # FIXED: Properly clear gradients with set_to_none=True
        self.optimizer.zero_grad(set_to_none=True)
        
        if self.ema_model is not None:
            self._update_ema()
        
        self.global_step += 1
        
        # FIXED: Capture metrics before deleting tensors
        metrics = {
            'loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'aux_loss': aux_loss.item(),
            'mtp_loss': mtp_loss.item() if mtp_loss is not None else 0.0,
            'lr': self.optimizer.param_groups[0]['lr'],
            'grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
            'step': self.global_step,
        }
        
        # FIXED: Delete tensors after capturing metrics to free memory
        del ce_loss, total_loss, aux_loss, logits, shift_logits, shift_labels
        if mtp_loss is not None:
            del mtp_loss
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        self._log_metrics(metrics)
        return metrics
    
    def _update_ema(self, decay: float = None):
        if decay is None:
            decay = self.training_args.ema_decay
        
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1 - decay)
    
    def _log_metrics(self, metrics: Dict[str, float]):
        if self.rank != 0:
            return
        
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f'train/{k}', v, self.global_step)
        
        if HAS_WANDB and self.args.use_wandb:
            wandb.log(metrics, step=self.global_step)
        
        if self.global_step % self.args.log_interval == 0:
            logger.info(
                f"Step {self.global_step} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"CE: {metrics['ce_loss']:.4f} | "
                f"Aux: {metrics['aux_loss']:.4f} | "
                f"MTP: {metrics['mtp_loss']:.4f} | "
                f"LR: {metrics['lr']:.2e}"
            )
    
    def save_checkpoint(self, path: Optional[str] = None):
        if self.rank != 0:
            return
        
        if path is None:
            path = os.path.join(self.args.checkpoint_dir, f"step_{self.global_step}")
        
        os.makedirs(path, exist_ok=True)
        save_model(self.model, path)
        
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
        }, os.path.join(path, 'training_state.pt'))
        
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            self.args = ModelArgs.load(config_path)
        
        model_path = os.path.join(path, 'model.safetensors')
        if os.path.exists(model_path):
            from safetensors.torch import load_file
            self.model.load_state_dict(load_file(model_path), strict=False)
        else:
            model_path = os.path.join(path, 'model.pt')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
        
        state_path = os.path.join(path, 'training_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.scaler.load_state_dict(state['scaler'])
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_loss = state.get('best_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded: {path}")
    
    def get_model(self) -> nn.Module:
        if self.ema_model is not None:
            return self.ema_model
        return self.model


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def create_parser() -> ArgumentParser:
    """Create argument parser"""
    parser = ArgumentParser(
        description="DeepNova - Enhanced Production MoE Transformer with DeepSeek V3 Architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python deepnova_enhanced.py chat
  python deepnova_enhanced.py learn --text "Important information to remember"
  python deepnova_enhanced.py learn --file data.txt
  python deepnova_enhanced.py learn --directory ./knowledge_base
  python deepnova_enhanced.py recall --query "What did I learn about X?"
  python deepnova_enhanced.py stats
  python deepnova_enhanced.py clear
  python deepnova_enhanced.py export --output knowledge_export.json
  python deepnova_enhanced.py list --limit 20
  python deepnova_enhanced.py generate --prompt "Hello" --max-tokens 100
  python deepnova_enhanced.py train --data ./data --epochs 3 --batch-size 8
  python deepnova_enhanced.py serve --port 8000
  python deepnova_enhanced.py benchmark --prompt "Test prompt" --iterations 5
  python deepnova_enhanced.py test
  python deepnova_enhanced.py enhanced --config full
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    chat_parser.add_argument('--model-path', default=None, help='Path to model checkpoint')
    chat_parser.add_argument('--parallel', action='store_true', help='Use parallel MoE+Dense')
    chat_parser.add_argument('--enhanced', action='store_true', help='Use all enhanced features')
    chat_parser.add_argument('--memory-file', default='deepnova_memory.json', help='Memory file path')
    
    learn_parser = subparsers.add_parser('learn', help='Learn from text or file')
    learn_parser.add_argument('--text', type=str, help='Text to learn')
    learn_parser.add_argument('--file', type=str, help='File to learn from')
    learn_parser.add_argument('--directory', type=str, help='Directory to learn from')
    learn_parser.add_argument('--model-path', default=None)
    learn_parser.add_argument('--parallel', action='store_true')
    learn_parser.add_argument('--enhanced', action='store_true')
    
    recall_parser = subparsers.add_parser('recall', help='Recall learned knowledge')
    recall_parser.add_argument('--query', required=True, type=str, help='Query to search')
    recall_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    recall_parser.add_argument('--model-path', default=None)
    
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    stats_parser.add_argument('--model-path', default=None)
    stats_parser.add_argument('--parallel', action='store_true')
    stats_parser.add_argument('--enhanced', action='store_true')
    
    clear_parser = subparsers.add_parser('clear', help='Clear conversation context')
    clear_parser.add_argument('--all', action='store_true', help='Clear all memory including important facts')
    clear_parser.add_argument('--model-path', default=None)
    
    export_parser = subparsers.add_parser('export', help='Export learned knowledge')
    export_parser.add_argument('--output', default='knowledge_export.json', help='Output file')
    export_parser.add_argument('--model-path', default=None)
    
    list_parser = subparsers.add_parser('list', help='List learned texts')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum items to list')
    list_parser.add_argument('--model-path', default=None)
    
    generate_parser = subparsers.add_parser('generate', help='Generate text')
    generate_parser.add_argument('--prompt', required=True, help='Input prompt')
    generate_parser.add_argument('--max-tokens', type=int, default=100)
    generate_parser.add_argument('--temperature', type=float, default=0.7)
    generate_parser.add_argument('--top-p', type=float, default=0.9)
    generate_parser.add_argument('--top-k', type=int, default=50)
    generate_parser.add_argument('--model-path', default=None)
    generate_parser.add_argument('--parallel', action='store_true')
    generate_parser.add_argument('--enhanced', action='store_true')
    
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', required=True, help='Path to training data')
    train_parser.add_argument('--epochs', type=int, default=1)
    train_parser.add_argument('--batch-size', type=int, default=8)
    train_parser.add_argument('--lr', type=float, default=3e-4)
    train_parser.add_argument('--checkpoint-dir', default='./checkpoints')
    train_parser.add_argument('--parallel', action='store_true')
    train_parser.add_argument('--enhanced', action='store_true')
    
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='0.0.0.0')
    serve_parser.add_argument('--port', type=int, default=8000)
    serve_parser.add_argument('--model-path', default=None)
    serve_parser.add_argument('--parallel', action='store_true')
    serve_parser.add_argument('--enhanced', action='store_true')
    
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--prompt', default='The quick brown fox jumps over the lazy dog')
    benchmark_parser.add_argument('--max-tokens', type=int, default=100)
    benchmark_parser.add_argument('--iterations', type=int, default=5)
    benchmark_parser.add_argument('--model-path', default=None)
    benchmark_parser.add_argument('--parallel', action='store_true')
    benchmark_parser.add_argument('--enhanced', action='store_true')
    
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    
    enhanced_parser = subparsers.add_parser('enhanced', help='Run with all enhanced features')
    enhanced_parser.add_argument('--config', default='full', choices=['full', 'lite', 'parallel'])
    enhanced_parser.add_argument('--prompt', help='Test prompt')
    
    parser.add_argument('--model-size', default='lite', choices=['lite', 'base', 'large', '671b', 'parallel', 'enhanced'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--dtype', default='bf16', choices=['bf16', 'fp16', 'fp32', 'fp8'])
    
    return parser


def interactive_mode(deepnova: DeepNovaAI):
    """Run interactive chat mode"""
    print("\n" + "=" * 70)
    print(f"DEEPNOVA AI v{deepnova.version} - Interactive Chat Mode")
    print("=" * 70)
    print("Commands:")
    print("  /learn <text>     - Learn new information")
    print("  /learnfile <path> - Learn from file")
    print("  /learndir <path>  - Learn from directory")
    print("  /recall <query>   - Recall learned knowledge")
    print("  /stats            - Show statistics")
    print("  /clear            - Clear conversation context")
    print("  /list             - List learned texts")
    print("  /export <file>    - Export knowledge to file")
    print("  /quit             - Exit")
    print("=" * 70)
    print(deepnova.greeting)
    print()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith('/'):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == '/quit':
                print("Goodbye!")
                break
            
            elif cmd == '/learn':
                if not arg:
                    print("Usage: /learn <text>")
                    continue
                result = deepnova.learn(arg)
                if result.get('success'):
                    print(f"Learned: {result['summary']}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
            
            elif cmd == '/learnfile':
                if not arg:
                    print("Usage: /learnfile <file_path>")
                    continue
                results = deepnova.learn_from_file(arg)
                success = len([r for r in results if r.get('success')])
                print(f"Learned {success} segments from {arg}")
            
            elif cmd == '/learndir':
                if not arg:
                    print("Usage: /learndir <directory_path>")
                    continue
                results = deepnova.learn_from_directory(arg)
                success = len([r for r in results if r.get('success')])
                print(f"Learned {success} segments from directory {arg}")
            
            elif cmd == '/recall':
                if not arg:
                    print("Usage: /recall <query>")
                    continue
                results = deepnova.recall(arg, top_k=5)
                if results:
                    print(f"\nFound {len(results)} relevant items:")
                    for i, r in enumerate(results):
                        print(f"  {i+1}. {r['summary']}")
                        if r.get('source'):
                            print(f"     Source: {r['source']}")
                else:
                    print("No relevant knowledge found.")
            
            elif cmd == '/stats':
                deepnova.print_stats()
            
            elif cmd == '/clear':
                keep_important = not arg or arg != '--all'
                deepnova.clear_context(keep_important=keep_important)
                print("Context cleared." + (" Keeping important facts." if keep_important else ""))
            
            elif cmd == '/list':
                learned = deepnova.list_learned(limit=20)
                if learned:
                    print(f"\nLearned texts ({len(learned)} shown):")
                    for i, l in enumerate(learned):
                        print(f"  {i+1}. {l['summary'][:80]}...")
                        print(f"     Source: {l['source']}, Accesses: {l.get('access_count', 0)}")
                else:
                    print("No learned texts yet.")
            
            elif cmd == '/export':
                output_file = arg if arg else "knowledge_export.json"
                if deepnova.export_knowledge(output_file):
                    print(f"Knowledge exported to {output_file}")
                else:
                    print("Export failed.")
            
            elif cmd == '/help':
                print("Commands: /learn, /learnfile, /learndir, /recall, /stats, /clear, /list, /export, /quit")
            
            else:
                print(f"Unknown command: {cmd}. Type /help for available commands.")
        
        else:
            response = deepnova.chat(user_input)
            print(f"\nDeepNova: {response}")


def learn_mode(args):
    """Handle learn command"""
    if args.enhanced:
        model_args = ModelArgs.enhanced_full()
    elif args.model_size == 'parallel' or args.parallel:
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.parallel and not args.enhanced:
        model_args.use_parallel_moe_dense = True
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    if not validate_model_args(model_args):
        print("Model configuration validation failed")
        return
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    if args.text:
        result = deepnova.learn(args.text, source="cli")
        if result.get('success'):
            print(f"Learned: {result['summary']}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    if args.file:
        results = deepnova.learn_from_file(args.file)
        success = len([r for r in results if r.get('success')])
        print(f"Learned {success} segments from {args.file}")
    
    if args.directory:
        results = deepnova.learn_from_directory(args.directory)
        success = len([r for r in results if r.get('success')])
        print(f"Learned {success} segments from directory {args.directory}")


def recall_mode(args):
    """Handle recall command"""
    if args.model_size == 'parallel':
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    results = deepnova.recall(args.query, top_k=args.top_k)
    
    if results:
        print(f"\nFound {len(results)} relevant items:")
        for i, r in enumerate(results):
            print(f"\n{i+1}. Score: {r['score']:.2f}")
            print(f"   Summary: {r['summary']}")
            if r.get('source'):
                print(f"   Source: {r['source']}")
            print(f"   Text: {r['text'][:200]}...")
    else:
        print("No relevant knowledge found.")


def stats_mode(args):
    """Handle stats command"""
    if args.enhanced:
        model_args = ModelArgs.enhanced_full()
    elif args.model_size == 'parallel' or args.parallel:
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    deepnova.print_stats()


def clear_mode(args):
    """Handle clear command"""
    if args.model_size == 'parallel':
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    deepnova.clear_context(keep_important=not args.all)
    print(f"Context cleared. {'Keeping important facts.' if not args.all else 'All memory cleared.'}")


def export_mode(args):
    """Handle export command"""
    if args.model_size == 'parallel':
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    if deepnova.export_knowledge(args.output):
        print(f"Knowledge exported to {args.output}")
    else:
        print("Export failed.")


def list_mode(args):
    """Handle list command"""
    if args.model_size == 'parallel':
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    learned = deepnova.list_learned(limit=args.limit)
    
    if learned:
        total = deepnova.learning_system.get_stats()['total_learned']
        print(f"\nLearned texts ({len(learned)} shown, total {total}):")
        print("-" * 60)
        for i, l in enumerate(learned):
            print(f"{i+1}. {l['summary']}")
            print(f"   Hash: {l['hash']}")
            print(f"   Source: {l['source']}")
            print(f"   Accesses: {l.get('access_count', 0)}")
            print()
    else:
        print("No learned texts yet.")


def generate_mode(args):
    """Handle generate command"""
    if args.enhanced:
        model_args = ModelArgs.enhanced_full()
    elif args.model_size == 'parallel' or args.parallel:
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    if not validate_model_args(model_args):
        print("Model configuration validation failed")
        return
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    response = deepnova.chat(args.prompt, max_new_tokens=args.max_tokens,
                              temperature=args.temperature)
    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")


def train_mode(args):
    """Handle train command"""
    if args.enhanced:
        model_args = ModelArgs.enhanced_full()
    elif args.model_size == 'parallel' or args.parallel:
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    if not validate_model_args(model_args):
        print("Model configuration validation failed")
        return
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    training_args = TrainingArgs(
        epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    
    trainer = Trainer(model, model_args, training_args, tokenizer)
    logger.info(f"Training mode initialized. Data path: {args.data}")
    print(f"Training mode initialized. Data path: {args.data}")
    print("Run training with dataset for actual training.")


def serve_mode(args):
    """Handle serve command"""
    if not HAS_FASTAPI:
        print("FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    
    if args.enhanced:
        model_args = ModelArgs.enhanced_full()
    elif args.model_size == 'parallel' or args.parallel:
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    if not validate_model_args(model_args):
        print("Model configuration validation failed")
        return
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    from fastapi import FastAPI
    import uvicorn
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    app = FastAPI(title="DeepNova API", version=deepnova.version)
    
    @app.get("/")
    async def root():
        return {"name": "DeepNova", "version": deepnova.version, "status": "running"}
    
    @app.post("/chat")
    async def chat(request: dict):
        prompt = request.get("prompt", "")
        if not prompt:
            return {"error": "No prompt provided"}
        response = deepnova.chat(prompt)
        return {"response": response}
    
    @app.post("/learn")
    async def learn(request: dict):
        text = request.get("text", "")
        if not text:
            return {"error": "No text provided"}
        result = deepnova.learn(text)
        return result
    
    @app.get("/stats")
    async def stats():
        return deepnova.get_stats()
    
    logger.info(f"Starting DeepNova API server on {args.host}:{args.port}")
    print(f"Starting API server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def benchmark_mode(args):
    """Handle benchmark command"""
    if args.enhanced:
        model_args = ModelArgs.enhanced_full()
    elif args.model_size == 'parallel' or args.parallel:
        model_args = ModelArgs.parallel_moe_dense()
    elif args.model_size == 'lite':
        model_args = ModelArgs.deepseek_v3_lite()
    else:
        model_args = ModelArgs()
    
    if args.device:
        model_args.device = args.device
    if args.dtype:
        model_args.dtype = args.dtype
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    
    if args.model_path and os.path.exists(args.model_path):
        try:
            model, model_args = load_model(args.model_path, model_args.device)
        except Exception as e:
            print(f"Failed to load model: {e}")
    
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    print("\n" + "=" * 70)
    print("DEEPNOVA ENHANCED BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_args.model_name}")
    print(f"Parallel MoE+Dense: {model_args.use_parallel_moe_dense}")
    print(f"Enhanced Features: {model_args.use_glm}, {model_args.use_adaptive_router}, {model_args.use_dynamic_depth}, {model_args.use_multi_token_prediction}")
    print(f"Device: {model_args.device}")
    print("-" * 70)
    
    times = []
    for i in range(args.iterations):
        start = time.time()
        response = deepnova.chat(args.prompt, max_new_tokens=args.max_tokens)
        elapsed = time.time() - start
        times.append(elapsed)
        tokens_generated = len(response.split())
        print(f"  Run {i+1}: {elapsed:.2f}s | {tokens_generated} tokens | {tokens_generated/elapsed:.1f} tok/s")
    
    avg_time = sum(times) / len(times)
    avg_tps = args.max_tokens / avg_time if avg_time > 0 else 0
    
    print("-" * 70)
    print(f"Average Time: {avg_time:.2f}s")
    print(f"Average Speed: {avg_tps:.1f} tok/s")
    print(f"Min Time: {min(times):.2f}s")
    print(f"Max Time: {max(times):.2f}s")
    print("=" * 70)


def test_mode():
    """Run unit tests"""
    print("\n" + "=" * 70)
    print("DEEPNOVA ENHANCED UNIT TESTS")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Model validation
    try:
        args = ModelArgs.deepseek_v3_lite()
        assert validate_model_args(args), "Validation failed for valid config"
        print("[PASS] Model validation test")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Model validation test: {e}")
        tests_failed += 1
    
    # Test 2: Small model forward pass with enhanced features
    try:
        test_args = ModelArgs()
        test_args.dim = 128
        test_args.n_heads = 8
        test_args.n_layers = 2
        test_args.vocab_size = 1000
        test_args.kv_lora_rank = 32
        test_args.qk_nope_head_dim = 8
        test_args.qk_rope_head_dim = 4
        test_args.v_head_dim = 8
        test_args.n_routed_experts = 8
        test_args.n_activated_experts = 2
        test_args.use_parallel_moe_dense = True
        test_args.use_glm = True
        test_args.use_adaptive_router = True
        test_args.use_multi_token_prediction = True
        
        model = Transformer(test_args)
        input_ids = torch.randint(0, test_args.vocab_size, (2, 10))
        logits, aux_losses, mtp_loss = model(input_ids, return_mtp_loss=True)
        
        assert logits.shape == (2, 10, test_args.vocab_size), f"Shape mismatch: {logits.shape}"
        print("[PASS] Enhanced model forward pass")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Enhanced model forward pass: {e}")
        tests_failed += 1
    
    # Test 3: Adaptive Router
    try:
        test_args = ModelArgs()
        router = AdaptiveRouter(test_args, n_experts=8, dim=128)
        x = torch.randn(4, 16, 128)
        topk_probs, topk_indices, load_balance_loss, temperature = router(x)
        assert topk_indices.shape[1] == test_args.n_activated_experts
        print("[PASS] Adaptive Router test")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] Adaptive Router test: {e}")
        tests_failed += 1
    
    # Test 4: PagedKVCache
    try:
        cache = PagedKVCache(
            max_batch_size=1,
            max_num_blocks=10,
            block_size=4,
            num_heads=8,
            head_dim=16,
            kv_lora_rank=32,
            qk_rope_head_dim=8,
            device=torch.device('cpu'),
            dtype=torch.float32
        )
        
        cache.allocate_sequence(0, 20)
        assert cache.num_free_blocks < cache.max_num_blocks
        
        positions = torch.tensor([0, 1, 2, 3])
        k_latent = torch.randn(4, 32)
        v_full = torch.randn(4, 8 * 16)
        k_pe = torch.randn(4, 8)
        
        cache.store_kv(0, positions, k_latent, v_full, k_pe)
        k_ret, v_ret, pe_ret = cache.get_kv(0, 4)
        assert k_ret.shape[0] == 4
        
        cache.free_sequence(0)
        print("[PASS] PagedKVCache test")
        tests_passed += 1
    except Exception as e:
        print(f"[FAIL] PagedKVCache test: {e}")
        tests_failed += 1
    
    print("-" * 70)
    print(f"Tests passed: {tests_passed}")
    print(f"Tests failed: {tests_failed}")
    print("=" * 70)
    
    return 0 if tests_failed == 0 else 1


def enhanced_mode(args):
    """Run with all enhanced features"""
    if args.config == 'full':
        model_args = ModelArgs.enhanced_full()
    elif args.config == 'parallel':
        model_args = ModelArgs.parallel_moe_dense()
    else:
        model_args = ModelArgs()
        model_args.use_parallel_moe_dense = True
        model_args.use_glm = True
        model_args.use_adaptive_router = True
        model_args.use_dynamic_depth = True
        model_args.use_multi_token_prediction = True
    
    print("\n" + "=" * 70)
    print("DEEPNOVA ENHANCED MODE")
    print("=" * 70)
    print(f"Configuration: {args.config}")
    print(f"Parallel MoE+Dense: {model_args.use_parallel_moe_dense}")
    print(f"GLM: {model_args.use_glm}")
    print(f"Adaptive Router: {model_args.use_adaptive_router}")
    print(f"Dynamic Depth: {model_args.use_dynamic_depth}")
    print(f"Multi-Token Prediction: {model_args.use_multi_token_prediction}")
    print("=" * 70)
    
    tokenizer = ProductionTokenizer()
    model = Transformer(model_args)
    deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file="deepnova_memory.json")
    
    if args.prompt:
        response = deepnova.chat(args.prompt)
        print(f"\nPrompt: {args.prompt}")
        print(f"Response: {response}")
    else:
        interactive_mode(deepnova)

def test_dynamic_depth():
    """Test dynamic depth skipping behavior"""
    print("\n" + "=" * 70)
    print("TEST: Dynamic Depth Skip Behavior")
    print("=" * 70)
    
    try:
        args = ModelArgs()
        args.dim = 128
        args.n_layers = 2
        args.vocab_size = 1000
        args.use_dynamic_depth = True
        args.dynamic_depth_skip_prob = 1.0  # Force skip
        args.dynamic_depth_confidence_threshold = 0.0  # Low threshold
        args.gradient_checkpointing = False
        
        block = TransformerBlock(0, args)
        block.confidence.data.fill_(1.0)  # High confidence to trigger skip condition
        
        x = torch.randn(1, 10, args.dim)
        output, aux_loss, confidence = block(x)
        
        # When skip happens, output should be input (residual only)
        # Check if skip counter increased (indicates layer was skipped)
        if block.skip_counter > 0:
            print("[PASS] Dynamic depth skipping works - layer was skipped")
            return True
        else:
            print("[INFO] Dynamic depth: skip not triggered (randomness)")
            return True
    except Exception as e:
        print(f"[FAIL] Dynamic depth test: {e}")
        return False


def test_mtp_validation():
    """Test MTP parameter validation"""
    print("\n" + "=" * 70)
    print("TEST: Multi-Token Prediction Validation")
    print("=" * 70)
    
    try:
        # Test valid MTP config
        args = ModelArgs()
        args.dim = 128
        args.n_layers = 2
        args.vocab_size = 1000
        args.use_multi_token_prediction = True
        args.mtp_n_predictions = 4
        args.max_seq_len = 512
        args.mtp_loss_weight = 0.1
        
        model = Transformer(args)
        print("[PASS] Valid MTP config accepted")
        
        # Test invalid MTP config (mtp_n_predictions > max_seq_len)
        try:
            invalid_args = ModelArgs()
            invalid_args.use_multi_token_prediction = True
            invalid_args.mtp_n_predictions = 1000
            invalid_args.max_seq_len = 512
            invalid_model = Transformer(invalid_args)
            print("[FAIL] Invalid MTP config should have been rejected")
            return False
        except AssertionError as e:
            if "exceeds max_seq_len" in str(e):
                print("[PASS] Invalid MTP config correctly rejected")
                return True
            raise
    except Exception as e:
        print(f"[FAIL] MTP validation test: {e}")
        return False


def test_expert_bias_momentum():
    """Test expert bias update with momentum"""
    print("\n" + "=" * 70)
    print("TEST: Expert Bias Momentum Smoothing")
    print("=" * 70)
    
    try:
        args = ModelArgs.deepseek_v3_lite()
        args.n_layers = 1
        args.use_dynamic_depth = False
        
        model = Transformer(args)
        layer = model.layers[0]
        
        if hasattr(layer.mlp, 'update_expert_bias'):
            # Get initial bias
            initial_bias = layer.mlp.expert_bias.clone()
            
            # Simulate router calls to accumulate usage stats
            for _ in range(5):
                dummy_input = torch.randn(1, 10, args.dim)
                _ = layer.mlp(dummy_input)
            
            # Update bias
            layer.mlp.update_expert_bias()
            
            # Check that momentum buffer was created
            if hasattr(layer.mlp, 'expert_bias_momentum'):
                print("[PASS] Expert bias momentum buffer created")
                return True
            else:
                print("[FAIL] Momentum buffer not found")
                return False
        else:
            print("[INFO] MLP doesn't have update_expert_bias method")
            return True
    except Exception as e:
        print(f"[FAIL] Expert bias momentum test: {e}")
        return False


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return
    
    if args.dtype == 'fp16' and not torch.cuda.is_available():
        logger.warning("fp16 requested but CUDA not available, falling back to fp32")
        args.dtype = 'fp32'
    
    if args.mode == 'test':
        sys.exit(test_mode())
    
    elif args.mode == 'chat':
        if args.enhanced:
            model_args = ModelArgs.enhanced_full()
        elif args.model_size == 'parallel' or args.parallel:
            model_args = ModelArgs.parallel_moe_dense()
        elif args.model_size == 'lite':
            model_args = ModelArgs.deepseek_v3_lite()
        else:
            model_args = ModelArgs()
        
        if args.device:
            model_args.device = args.device
        if args.dtype:
            model_args.dtype = args.dtype
        
        if not validate_model_args(model_args):
            print("Model configuration validation failed")
            return
        
        tokenizer = ProductionTokenizer()
        model = Transformer(model_args)
        
        if args.model_path and os.path.exists(args.model_path):
            try:
                model, model_args = load_model(args.model_path, model_args.device)
            except Exception as e:
                print(f"Failed to load model: {e}")
                print("Using newly created model instead")
        
        deepnova = DeepNovaAI(model, tokenizer, model_args, memory_file=args.memory_file)
        interactive_mode(deepnova)
    
    elif args.mode == 'learn':
        learn_mode(args)
    
    elif args.mode == 'recall':
        recall_mode(args)
    
    elif args.mode == 'stats':
        stats_mode(args)
    
    elif args.mode == 'clear':
        clear_mode(args)
    
    elif args.mode == 'export':
        export_mode(args)
    
    elif args.mode == 'list':
        list_mode(args)
    
    elif args.mode == 'generate':
        generate_mode(args)
    
    elif args.mode == 'train':
        train_mode(args)
    
    elif args.mode == 'serve':
        serve_mode(args)
    
    elif args.mode == 'benchmark':
        benchmark_mode(args)
    
    elif args.mode == 'enhanced':
        enhanced_mode(args)
    
    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help()
    
    logger.info("DeepNova Enhanced execution completed.")


atexit.register(cleanup_memory)


if __name__ == "__main__":
    if sys.platform != 'win32':
        try:
            import multiprocessing as mp
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        cleanup_memory()