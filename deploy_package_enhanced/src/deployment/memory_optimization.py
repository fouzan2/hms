"""
Memory and Computational Optimization for HMS EEG Classification

This module implements memory and computational optimization techniques:
- Gradient checkpointing for memory-efficient training
- Mixed precision training (FP16/BF16)
- Memory mapping for large datasets
- Data pipeline optimization
- CPU-GPU memory transfer optimization
- Memory profiling and monitoring
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from pathlib import Path
import psutil
import gc
import logging
import time
from dataclasses import dataclass
import mmap
import h5py
from contextlib import contextmanager
import threading
from queue import Queue
import warnings

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    enable_gradient_checkpointing: bool = True
    gradient_checkpoint_segments: int = 4
    enable_mixed_precision: bool = True
    mixed_precision_dtype: str = "fp16"  # fp16, bf16
    enable_cpu_offload: bool = False
    memory_map_threshold: int = 100 * 1024 * 1024  # 100MB
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    max_memory_cache: int = 8 * 1024 * 1024 * 1024  # 8GB
    enable_memory_profiling: bool = True
    lazy_loading: bool = True


class GradientCheckpointer:
    """Implements gradient checkpointing for memory-efficient training."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        
    def apply_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to model layers."""
        if not self.config.enable_gradient_checkpointing:
            return model
            
        logger.info("Applying gradient checkpointing...")
        
        # Find checkpointable segments
        checkpointable_modules = self._find_checkpointable_modules(model)
        
        # Wrap forward methods
        for module in checkpointable_modules:
            original_forward = module.forward
            module.forward = self._create_checkpointed_forward(original_forward)
            
        logger.info(f"Gradient checkpointing applied to {len(checkpointable_modules)} modules")
        return model
    
    def _find_checkpointable_modules(self, model: nn.Module) -> List[nn.Module]:
        """Find modules suitable for checkpointing."""
        checkpointable = []
        
        # Common patterns for checkpointing
        checkpoint_types = (
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            # Add ResNet blocks, etc. based on your architecture
        )
        
        for name, module in model.named_modules():
            # Check if module is a known checkpointable type
            if isinstance(module, checkpoint_types):
                checkpointable.append(module)
            # Check for custom blocks with specific patterns
            elif hasattr(module, 'is_checkpointable') and module.is_checkpointable:
                checkpointable.append(module)
                
        # If no specific modules found, segment the model
        if not checkpointable:
            checkpointable = self._segment_model(model)
            
        return checkpointable
    
    def _segment_model(self, model: nn.Module) -> List[nn.Module]:
        """Segment model into checkpointable parts."""
        segments = []
        modules = list(model.children())
        
        if len(modules) >= self.config.gradient_checkpoint_segments:
            # Divide into segments
            segment_size = len(modules) // self.config.gradient_checkpoint_segments
            for i in range(0, len(modules), segment_size):
                segment = nn.Sequential(*modules[i:i+segment_size])
                segments.append(segment)
        else:
            # Use individual modules
            segments = modules
            
        return segments
    
    def _create_checkpointed_forward(self, forward_func: Callable) -> Callable:
        """Create a checkpointed version of forward function."""
        def checkpointed_forward(*args, **kwargs):
            # Use checkpoint if training
            if args[0].training:  # args[0] is self
                return checkpoint(forward_func, *args, **kwargs)
            else:
                return forward_func(*args, **kwargs)
        return checkpointed_forward


class MixedPrecisionOptimizer:
    """Implements mixed precision training for faster computation."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.scaler = None
        self.setup()
        
    def setup(self):
        """Setup mixed precision training."""
        if not self.config.enable_mixed_precision:
            return
            
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Mixed precision disabled.")
            self.config.enable_mixed_precision = False
            return
            
        # Check dtype support
        if self.config.mixed_precision_dtype == "bf16":
            if not torch.cuda.is_bf16_supported():
                logger.warning("BF16 not supported. Falling back to FP16.")
                self.config.mixed_precision_dtype = "fp16"
                
        # Create gradient scaler for FP16
        if self.config.mixed_precision_dtype == "fp16":
            self.scaler = GradScaler()
            
        logger.info(f"Mixed precision training enabled with {self.config.mixed_precision_dtype}")
    
    def get_autocast_context(self):
        """Get autocast context manager."""
        if not self.config.enable_mixed_precision:
            return contextmanager(lambda: iter([None]))()
            
        dtype = torch.float16 if self.config.mixed_precision_dtype == "fp16" else torch.bfloat16
        return autocast(dtype=dtype)
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        if self.scaler and self.config.mixed_precision_dtype == "fp16":
            return self.scaler.scale(loss)
        return loss
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer):
        """Step optimizer with mixed precision handling."""
        if self.scaler and self.config.mixed_precision_dtype == "fp16":
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients for gradient clipping."""
        if self.scaler and self.config.mixed_precision_dtype == "fp16":
            self.scaler.unscale_(optimizer)


class MemoryMappedDataset:
    """Memory-mapped dataset for efficient large file handling."""
    
    def __init__(self, data_path: Path, config: MemoryConfig):
        self.data_path = data_path
        self.config = config
        self.file_handles = {}
        self.mmap_handles = {}
        self._setup_memory_mapping()
        
    def _setup_memory_mapping(self):
        """Setup memory mapping for large files."""
        # Check if HDF5 file
        if self.data_path.suffix == '.h5':
            self.data_format = 'hdf5'
            self.h5_file = h5py.File(self.data_path, 'r', swmr=True)
        # Check if NumPy memmap
        elif self.data_path.suffix == '.npy':
            self.data_format = 'numpy'
            self.mmap_array = np.load(self.data_path, mmap_mode='r')
        # Raw binary file
        else:
            self.data_format = 'raw'
            self._setup_raw_mmap()
    
    def _setup_raw_mmap(self):
        """Setup memory mapping for raw binary files."""
        file_size = self.data_path.stat().st_size
        
        if file_size > self.config.memory_map_threshold:
            # Use memory mapping
            with open(self.data_path, 'r+b') as f:
                self.mmap_handles[str(self.data_path)] = mmap.mmap(
                    f.fileno(), 0, access=mmap.ACCESS_READ
                )
        else:
            # Load into memory
            with open(self.data_path, 'rb') as f:
                self.data = f.read()
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """Get item with lazy loading."""
        if self.data_format == 'hdf5':
            # Lazy load from HDF5
            return self.h5_file['data'][idx]
        elif self.data_format == 'numpy':
            # Use memory-mapped array
            return self.mmap_array[idx]
        else:
            # Handle raw format based on your data structure
            raise NotImplementedError("Raw format getitem not implemented")
    
    def __del__(self):
        """Cleanup memory mappings."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()
        for mmap_handle in self.mmap_handles.values():
            mmap_handle.close()


class DataPipelineOptimizer:
    """Optimizes data loading pipeline for maximum throughput."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.prefetch_queue = Queue(maxsize=config.prefetch_factor)
        self.prefetch_thread = None
        
    def create_optimized_dataloader(self, dataset: torch.utils.data.Dataset,
                                  batch_size: int, shuffle: bool = True,
                                  **kwargs) -> torch.utils.data.DataLoader:
        """Create optimized DataLoader with all performance features."""
        
        # Determine optimal number of workers
        num_workers = self._determine_optimal_workers()
        
        # Create DataLoader with optimizations
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.config.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.config.persistent_workers and num_workers > 0,
            prefetch_factor=self.config.prefetch_factor if num_workers > 0 else 2,
            **kwargs
        )
        
        # Wrap with additional prefetching if needed
        if torch.cuda.is_available():
            dataloader = self._add_cuda_prefetching(dataloader)
            
        return dataloader
    
    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers based on system resources."""
        cpu_count = psutil.cpu_count(logical=False)
        
        # Use configured value if reasonable
        if 0 < self.config.num_workers <= cpu_count:
            return self.config.num_workers
            
        # Otherwise, use heuristic
        optimal_workers = min(cpu_count, 8)  # Cap at 8 workers
        
        # Reduce if memory constrained
        available_memory = psutil.virtual_memory().available
        if available_memory < 8 * 1024 * 1024 * 1024:  # Less than 8GB
            optimal_workers = min(optimal_workers, 4)
            
        return optimal_workers
    
    def _add_cuda_prefetching(self, dataloader: torch.utils.data.DataLoader):
        """Add CUDA stream prefetching to DataLoader."""
        
        class CUDAPrefetchDataLoader:
            def __init__(self, dataloader, device='cuda'):
                self.dataloader = dataloader
                self.device = device
                self.stream = torch.cuda.Stream()
                
            def __iter__(self):
                first = True
                for next_data in self.dataloader:
                    with torch.cuda.stream(self.stream):
                        # Transfer data to GPU asynchronously
                        next_data = self._to_device(next_data)
                        
                    if not first:
                        yield current_data
                    else:
                        first = False
                        
                    torch.cuda.current_stream().wait_stream(self.stream)
                    current_data = next_data
                    
                yield current_data
                
            def _to_device(self, data):
                if isinstance(data, torch.Tensor):
                    return data.to(self.device, non_blocking=True)
                elif isinstance(data, (list, tuple)):
                    return type(data)(self._to_device(d) for d in data)
                elif isinstance(data, dict):
                    return {k: self._to_device(v) for k, v in data.items()}
                else:
                    return data
                    
            def __len__(self):
                return len(self.dataloader)
                
        return CUDAPrefetchDataLoader(dataloader)


class MemoryOptimizer:
    """Central memory optimization manager."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_cache = {}
        self.cache_size = 0
        self.profiling_enabled = config.enable_memory_profiling
        
    def optimize_model_memory(self, model: nn.Module) -> nn.Module:
        """Apply all memory optimizations to model."""
        # Enable gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            checkpointer = GradientCheckpointer(self.config)
            model = checkpointer.apply_gradient_checkpointing(model)
            
        # Enable CPU offloading for large models
        if self.config.enable_cpu_offload:
            model = self._enable_cpu_offload(model)
            
        # Optimize buffer allocation
        self._optimize_buffers(model)
        
        return model
    
    def _enable_cpu_offload(self, model: nn.Module) -> nn.Module:
        """Enable CPU offloading for model parameters."""
        # This is a simplified version - for production use libraries like FairScale
        logger.info("Enabling CPU offload for large model parameters...")
        
        for name, param in model.named_parameters():
            if param.numel() > 10_000_000:  # Offload parameters > 10M
                # Mark for CPU offload
                param.data = param.data.cpu()
                param._cpu_offload = True
                
        return model
    
    def _optimize_buffers(self, model: nn.Module):
        """Optimize model buffer allocation."""
        # Share buffers where possible
        buffer_dict = {}
        
        for name, buffer in model.named_buffers():
            buffer_key = (buffer.shape, buffer.dtype)
            
            if buffer_key in buffer_dict:
                # Reuse existing buffer
                parent_name, parent_attr = name.rsplit('.', 1)
                parent_module = model
                for part in parent_name.split('.'):
                    parent_module = getattr(parent_module, part)
                setattr(parent_module, parent_attr, buffer_dict[buffer_key])
            else:
                buffer_dict[buffer_key] = buffer
    
    @contextmanager
    def memory_efficient_inference(self, model: nn.Module):
        """Context manager for memory-efficient inference."""
        # Disable gradient computation
        with torch.no_grad():
            # Enable eval mode
            was_training = model.training
            model.eval()
            
            # Clear cache before inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            try:
                yield model
            finally:
                # Restore training mode
                if was_training:
                    model.train()
                    
                # Clear cache after inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # Force garbage collection
                gc.collect()
    
    def profile_memory_usage(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict]:
        """Profile memory usage of a function."""
        if not self.profiling_enabled:
            return func(*args, **kwargs), {}
            
        # Get initial memory state
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu = torch.cuda.memory_allocated()
        
        initial_cpu = psutil.Process().memory_info().rss
        
        # Run function
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Get final memory state
        final_cpu = psutil.Process().memory_info().rss
        
        stats = {
            'execution_time': end_time - start_time,
            'cpu_memory_used': (final_cpu - initial_cpu) / 1024 / 1024,  # MB
        }
        
        if torch.cuda.is_available():
            final_gpu = torch.cuda.memory_allocated()
            peak_gpu = torch.cuda.max_memory_allocated()
            
            stats.update({
                'gpu_memory_used': (final_gpu - initial_gpu) / 1024 / 1024,  # MB
                'gpu_memory_peak': peak_gpu / 1024 / 1024,  # MB
            })
            
        return result, stats
    
    def optimize_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...],
                          target_memory_usage: float = 0.9) -> int:
        """Find optimal batch size for given memory constraints."""
        if not torch.cuda.is_available():
            return 32  # Default for CPU
            
        # Binary search for optimal batch size
        min_batch = 1
        max_batch = 256
        optimal_batch = 1
        
        while min_batch <= max_batch:
            batch_size = (min_batch + max_batch) // 2
            
            try:
                # Test with dummy input
                dummy_input = torch.randn(batch_size, *input_shape).cuda()
                
                with self.memory_efficient_inference(model):
                    _ = model(dummy_input)
                    
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                
                if memory_used < target_memory_usage:
                    optimal_batch = batch_size
                    min_batch = batch_size + 1
                else:
                    max_batch = batch_size - 1
                    
                # Clear memory
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    max_batch = batch_size - 1
                    torch.cuda.empty_cache()
                else:
                    raise
                    
        logger.info(f"Optimal batch size: {optimal_batch}")
        return optimal_batch


class LazyTensor:
    """Lazy loading tensor wrapper for delayed computation."""
    
    def __init__(self, load_func: Callable[[], torch.Tensor]):
        self.load_func = load_func
        self._tensor = None
        
    def _ensure_loaded(self):
        """Load tensor if not already loaded."""
        if self._tensor is None:
            self._tensor = self.load_func()
            
    def __getattr__(self, name):
        """Delegate attribute access to underlying tensor."""
        self._ensure_loaded()
        return getattr(self._tensor, name)
        
    def __getitem__(self, key):
        """Delegate indexing to underlying tensor."""
        self._ensure_loaded()
        return self._tensor[key]
        
    def to(self, *args, **kwargs):
        """Move tensor to device."""
        self._ensure_loaded()
        self._tensor = self._tensor.to(*args, **kwargs)
        return self


def create_memory_optimized_training_loop(model: nn.Module, config: MemoryConfig):
    """Create memory-optimized training loop."""
    
    # Initialize optimizers
    memory_optimizer = MemoryOptimizer(config)
    mixed_precision = MixedPrecisionOptimizer(config)
    
    # Optimize model
    model = memory_optimizer.optimize_model_memory(model)
    
    def training_step(batch, optimizer):
        """Single training step with memory optimization."""
        
        # Use mixed precision
        with mixed_precision.get_autocast_context():
            outputs = model(batch['input'])
            loss = compute_loss(outputs, batch['target'])
            
        # Scale loss and backward
        scaled_loss = mixed_precision.scale_loss(loss)
        scaled_loss.backward()
        
        # Unscale gradients for clipping
        mixed_precision.unscale_gradients(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step optimizer
        mixed_precision.step_optimizer(optimizer)
        optimizer.zero_grad()
        
        return loss.item()
    
    return training_step 