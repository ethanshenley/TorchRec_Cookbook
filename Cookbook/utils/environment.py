import torch
import os
from typing import Dict, Any

class TorchRecEnvironment:
    """Environment validation for TorchRec"""
    
    @staticmethod
    def check_environment() -> Dict[str, bool]:
        """Validate environment setup"""
        return {
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "distributed_available": torch.distributed.is_available(),
            "fbgemm_available": True if "fbgemm" in globals() else False
        }
    
    @staticmethod
    def check_distributed_setup() -> Dict[str, Any]:
        """Validate distributed environment"""
        required_vars = ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"]
        return {
            "env_vars_set": all(var in os.environ for var in required_vars),
            "process_group_initialized": torch.distributed.is_initialized(),
            "backend": torch.distributed.get_backend() if torch.distributed.is_initialized() else None
        }