import torch
import gc
from typing import Dict, Any

class TorchRecDebugger:
    """Debugging utilities for TorchRec"""
    
    @staticmethod
    def memory_status() -> Dict[str, Any]:
        """Get current GPU memory status"""
        return {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }
    
    @staticmethod
    def clear_memory():
        """Clear GPU cache and garbage collect"""
        gc.collect()
        torch.cuda.empty_cache()
        
    @staticmethod
    def validate_kjt(kjt) -> Dict[str, bool]:
        """Validate KeyedJaggedTensor structure"""
        return {
            "valid_keys": len(kjt.keys()) > 0,
            "lengths_match": kjt.lengths().sum() == len(kjt.values()),
            "on_device": kjt.values().is_cuda if torch.cuda.is_available() else True
        }
    
    @staticmethod
    def check_sharding(model) -> Dict[str, Any]:
        """Check sharding configuration"""
        return {
            "world_size": torch.distributed.get_world_size() 
                         if torch.distributed.is_initialized() else 1,
            "local_rank": torch.distributed.get_rank() 
                         if torch.distributed.is_initialized() else 0,
            "device_count": torch.cuda.device_count()
        }