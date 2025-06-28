"""
Distributed Training and Serving for HMS EEG Classification

This module implements distributed training and serving capabilities:
- Data parallelism with DDP
- Model parallelism for large models
- Parameter server architecture
- Gradient compression
- Fault-tolerant training
- Distributed model serving
- Load balancing
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
from pathlib import Path
import logging
import time
import os
import socket
import pickle
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from queue import Queue

# Optional dependencies
try:
    import horovod.torch as hvd
    HOROVOD_AVAILABLE = True
except ImportError:
    hvd = None
    HOROVOD_AVAILABLE = False

import ray
from ray import train, tune
from ray.train.torch import TorchTrainer
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = -1  # -1 for auto-detect
    master_addr: str = "localhost"
    master_port: str = "29500"
    init_method: str = "env://"
    enable_gradient_compression: bool = True
    compression_ratio: float = 0.1
    enable_fault_tolerance: bool = True
    checkpoint_interval: int = 100
    enable_model_parallel: bool = False
    pipeline_parallel_size: int = 2
    tensor_parallel_size: int = 2
    use_zero_optimizer: bool = True
    gradient_accumulation_steps: int = 1
    sync_batch_norm: bool = True
    find_unused_parameters: bool = False


class DistributedTrainingManager:
    """Manages distributed training setup and execution."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.is_initialized = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        
    def initialize(self):
        """Initialize distributed training environment."""
        if self.is_initialized:
            return
            
        # Setup based on backend
        if self.config.backend == "horovod":
            self._init_horovod()
        elif self.config.backend in ["nccl", "gloo"]:
            self._init_pytorch_dist()
        elif self.config.backend == "ray":
            self._init_ray()
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
            
        self.is_initialized = True
        logger.info(f"Distributed training initialized: rank={self.rank}, world_size={self.world_size}")
    
    def _init_pytorch_dist(self):
        """Initialize PyTorch distributed training."""
        # Set environment variables if not already set
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = self.config.master_addr
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = self.config.master_port
            
        # Initialize process group
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size if self.config.world_size > 0 else None,
            rank=int(os.environ.get("RANK", 0))
        )
        
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size()
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
    
    def _init_horovod(self):
        """Initialize Horovod distributed training."""
        if not HOROVOD_AVAILABLE:
            raise ImportError(
                "Horovod is not available. Please install horovod or use a different backend "
                "like 'nccl' or 'gloo' for PyTorch DDP. "
                "To install horovod: pip install horovod"
            )
        
        hvd.init()
        self.rank = hvd.rank()
        self.local_rank = hvd.local_rank()
        self.world_size = hvd.size()
        
        # Set GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
    
    def _init_ray(self):
        """Initialize Ray distributed training."""
        if not ray.is_initialized():
            ray.init()
        
        # Ray handles rank/world_size internally
        self.rank = 0  # Will be set by Ray
        self.world_size = 1  # Will be set by Ray
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.config.backend in ["nccl", "gloo"] and dist.is_initialized():
            dist.destroy_process_group()
        elif self.config.backend == "ray" and ray.is_initialized():
            ray.shutdown()
            
        self.is_initialized = False


class DataParallelTrainer:
    """Implements data parallel training with DDP."""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.config = config
        self.model = model
        self.ddp_model = None
        
    def setup_ddp(self, device_ids: Optional[List[int]] = None) -> nn.Module:
        """Setup DistributedDataParallel wrapper."""
        if not dist.is_initialized():
            raise RuntimeError("Distributed not initialized")
            
        # Move model to device
        device = torch.device(f"cuda:{dist.get_rank() % torch.cuda.device_count()}")
        self.model = self.model.to(device)
        
        # Sync batch norm if enabled
        if self.config.sync_batch_norm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        # Wrap with DDP
        self.ddp_model = DDP(
            self.model,
            device_ids=device_ids or [device.index],
            find_unused_parameters=self.config.find_unused_parameters
        )
        
        return self.ddp_model
    
    def train_step(self, batch: Dict[str, torch.Tensor], 
                   optimizer: torch.optim.Optimizer,
                   criterion: nn.Module,
                   scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
        """Single training step with DDP."""
        # Forward pass
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = self.ddp_model(batch['input'])
                loss = criterion(outputs, batch['target'])
        else:
            outputs = self.ddp_model(batch['input'])
            loss = criterion(outputs, batch['target'])
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (batch['step'] + 1) % self.config.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        return loss.item()


class ModelParallelTrainer:
    """Implements model parallelism for very large models."""
    
    def __init__(self, model: nn.Module, config: DistributedConfig):
        self.config = config
        self.model = model
        self.model_parallel_size = config.tensor_parallel_size
        self.pipeline_parallel_size = config.pipeline_parallel_size
        
    def setup_model_parallel(self) -> nn.Module:
        """Setup model parallelism."""
        # This is a simplified example - real implementation would use
        # libraries like FairScale or DeepSpeed
        
        if self.config.tensor_parallel_size > 1:
            self.model = self._setup_tensor_parallel(self.model)
            
        if self.config.pipeline_parallel_size > 1:
            self.model = self._setup_pipeline_parallel(self.model)
            
        return self.model
    
    def _setup_tensor_parallel(self, model: nn.Module) -> nn.Module:
        """Setup tensor parallelism by splitting layers."""
        # Example: Split linear layers across GPUs
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Replace with column parallel linear
                in_features = module.in_features
                out_features = module.out_features // self.model_parallel_size
                
                # Create new parallel linear layer
                new_module = ColumnParallelLinear(
                    in_features, out_features,
                    bias=module.bias is not None
                )
                
                # Replace in model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, new_module)
                
        return model
    
    def _setup_pipeline_parallel(self, model: nn.Module) -> nn.Module:
        """Setup pipeline parallelism by splitting model stages."""
        # Split model into stages
        stages = self._split_model_stages(model)
        
        # Wrap with pipeline parallel
        from torch.distributed.pipeline.sync import Pipe
        
        model = Pipe(
            nn.Sequential(*stages),
            balance=[1] * len(stages),  # Equal split
            devices=list(range(self.pipeline_parallel_size)),
            chunks=8  # Micro-batches
        )
        
        return model
    
    def _split_model_stages(self, model: nn.Module) -> List[nn.Module]:
        """Split model into pipeline stages."""
        # This is model-specific
        modules = list(model.children())
        stage_size = len(modules) // self.pipeline_parallel_size
        
        stages = []
        for i in range(0, len(modules), stage_size):
            stage = nn.Sequential(*modules[i:i+stage_size])
            stages.append(stage)
            
        return stages


class ColumnParallelLinear(nn.Module):
    """Column parallel linear layer for tensor parallelism."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Get parallel info
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        
        # Create local weight shard
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with column parallelism."""
        # Local computation
        output = torch.matmul(input, self.weight.t())
        
        if self.bias is not None:
            output = output + self.bias
            
        # No reduction needed for column parallel
        return output


class GradientCompressor:
    """Implements gradient compression for efficient communication."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.compression_ratio = config.compression_ratio
        self.error_feedback = {}
        
    def compress_gradient(self, grad: torch.Tensor, name: str) -> torch.Tensor:
        """Compress gradient using sparsification or quantization."""
        if not self.config.enable_gradient_compression:
            return grad
            
        # Add error feedback if available
        if name in self.error_feedback:
            grad = grad + self.error_feedback[name]
            
        # Sparsification
        if self.compression_ratio < 1.0:
            compressed_grad, indices = self._sparsify(grad)
            
            # Calculate error for feedback
            decompressed = torch.zeros_like(grad)
            decompressed.view(-1)[indices] = compressed_grad
            self.error_feedback[name] = grad - decompressed
            
            return decompressed
        else:
            return grad
    
    def _sparsify(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sparsify tensor by keeping top-k elements."""
        tensor_flat = tensor.view(-1)
        k = max(1, int(tensor_flat.numel() * self.compression_ratio))
        
        # Get top-k values and indices
        values, indices = torch.topk(tensor_flat.abs(), k)
        values = tensor_flat[indices]
        
        return values, indices
    
    def _quantize(self, tensor: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
        """Quantize tensor to reduce communication."""
        # Min-max quantization
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Quantize
        scale = (max_val - min_val) / (2**num_bits - 1)
        quantized = torch.round((tensor - min_val) / scale)
        
        # Dequantize
        dequantized = quantized * scale + min_val
        
        return dequantized


class FaultTolerantTrainer:
    """Implements fault-tolerant distributed training."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.checkpoint_dir = Path("checkpoints/distributed")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, model: nn.Module, 
                       optimizer: torch.optim.Optimizer,
                       metrics: Dict[str, float]):
        """Save training checkpoint for fault tolerance."""
        if dist.get_rank() != 0:
            return  # Only rank 0 saves
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save with atomic write
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)
        
        # Keep only recent checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, model: nn.Module, 
                       optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Load latest checkpoint for resuming training."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if not checkpoints:
            logger.info("No checkpoint found, starting from scratch")
            return {'epoch': 0, 'metrics': {}}
            
        latest_checkpoint = checkpoints[-1]
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'metrics': checkpoint.get('metrics', {})
        }
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoints to save space."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint.unlink()
    
    @contextmanager
    def fault_tolerant_execution(self):
        """Context manager for fault-tolerant execution."""
        try:
            yield
        except Exception as e:
            logger.error(f"Error during training: {e}")
            
            # Save emergency checkpoint
            if hasattr(self, 'model') and hasattr(self, 'optimizer'):
                self.save_checkpoint(-1, self.model, self.optimizer, {})
                
            # Re-raise for handling
            raise


class DistributedModelServer:
    """Distributed model serving with load balancing."""
    
    def __init__(self, model_path: Path, config: DistributedConfig):
        self.model_path = model_path
        self.config = config
        self.models = {}
        self.load_balancer = LoadBalancer()
        
    async def setup_serving(self, num_replicas: int = 4):
        """Setup distributed model serving."""
        # Load model replicas
        for i in range(num_replicas):
            device = f"cuda:{i % torch.cuda.device_count()}"
            model = self._load_model(device)
            self.models[i] = {
                'model': model,
                'device': device,
                'requests': 0
            }
            
        # Start model servers
        await self._start_model_servers()
        
    def _load_model(self, device: str) -> nn.Module:
        """Load model on specified device."""
        model = torch.load(self.model_path, map_location=device)
        model.eval()
        return model
    
    async def _start_model_servers(self):
        """Start async model servers."""
        tasks = []
        for replica_id, model_info in self.models.items():
            task = asyncio.create_task(
                self._run_model_server(replica_id, model_info)
            )
            tasks.append(task)
            
        await asyncio.gather(*tasks)
    
    async def _run_model_server(self, replica_id: int, model_info: Dict):
        """Run individual model server."""
        model = model_info['model']
        device = model_info['device']
        
        async def process_request(request):
            """Process single inference request."""
            input_data = torch.tensor(request['data']).to(device)
            
            with torch.no_grad():
                output = model(input_data)
                
            return output.cpu().numpy()
        
        # Server loop
        while True:
            request = await self.load_balancer.get_request()
            if request is None:
                break
                
            result = await process_request(request)
            await self.load_balancer.send_response(request['id'], result)
            
            model_info['requests'] += 1
    
    async def predict(self, data: np.ndarray) -> np.ndarray:
        """Make prediction using load balancing."""
        request_id = str(time.time())
        request = {
            'id': request_id,
            'data': data
        }
        
        # Submit to load balancer
        await self.load_balancer.submit_request(request)
        
        # Wait for response
        response = await self.load_balancer.wait_for_response(request_id)
        
        return response


class LoadBalancer:
    """Load balancer for distributed serving."""
    
    def __init__(self):
        self.request_queue = asyncio.Queue()
        self.response_map = {}
        self.pending_responses = {}
        
    async def submit_request(self, request: Dict):
        """Submit request to queue."""
        await self.request_queue.put(request)
        self.pending_responses[request['id']] = asyncio.Event()
        
    async def get_request(self) -> Optional[Dict]:
        """Get next request from queue."""
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
            
    async def send_response(self, request_id: str, response: Any):
        """Send response for request."""
        self.response_map[request_id] = response
        if request_id in self.pending_responses:
            self.pending_responses[request_id].set()
            
    async def wait_for_response(self, request_id: str) -> Any:
        """Wait for response for specific request."""
        if request_id in self.pending_responses:
            await self.pending_responses[request_id].wait()
            response = self.response_map.pop(request_id)
            del self.pending_responses[request_id]
            return response
        else:
            raise ValueError(f"Unknown request ID: {request_id}")


def create_distributed_trainer(model: nn.Module, config: DistributedConfig) -> Dict[str, Any]:
    """Create distributed trainer with all components."""
    
    # Initialize distributed environment
    dist_manager = DistributedTrainingManager(config)
    dist_manager.initialize()
    
    # Setup data parallel
    dp_trainer = DataParallelTrainer(model, config)
    ddp_model = dp_trainer.setup_ddp()
    
    # Setup gradient compression
    grad_compressor = GradientCompressor(config)
    
    # Setup fault tolerance
    fault_tolerant = FaultTolerantTrainer(config)
    
    # Create optimizer with ZeRO if enabled
    if config.use_zero_optimizer:
        optimizer = ZeroRedundancyOptimizer(
            ddp_model.parameters(),
            optimizer_class=torch.optim.AdamW,
            lr=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
    
    # Register gradient compression hooks
    if config.enable_gradient_compression:
        for name, param in ddp_model.named_parameters():
            if param.requires_grad:
                param.register_hook(
                    lambda grad, name=name: grad_compressor.compress_gradient(grad, name)
                )
    
    return {
        'model': ddp_model,
        'optimizer': optimizer,
        'dist_manager': dist_manager,
        'dp_trainer': dp_trainer,
        'fault_tolerant': fault_tolerant,
        'grad_compressor': grad_compressor
    } 