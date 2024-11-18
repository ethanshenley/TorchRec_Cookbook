import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Configuration for synthetic data generation"""
    num_users: int = 1000
    num_products: int = 10000
    max_sequence_length: int = 20
    min_sequence_length: int = 1
    batch_size: int = 32
    seed: Optional[int] = None

class TorchRecDataGenerator:
    """Generate synthetic data for TorchRec examples"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        if config.seed is not None:
            torch.manual_seed(config.seed)
            
    def generate_sequence_lengths(self, batch_size: int) -> torch.Tensor:
        """Generate random sequence lengths for batch"""
        return torch.randint(
            self.config.min_sequence_length,
            self.config.max_sequence_length,
            (batch_size,)
        )
    
    def generate_batch(self) -> Dict[str, torch.Tensor]:
        """Generate a batch of synthetic data"""
        lengths = self.generate_sequence_lengths(self.config.batch_size)
        total_values = lengths.sum().item()
        
        return {
            "values": torch.randint(0, self.config.num_products, (total_values,)),
            "lengths": lengths
        }
    
    def generate_kjt_inputs(self, feature_names: List[str]) -> Dict[str, torch.Tensor]:
        """Generate inputs suitable for KeyedJaggedTensor"""
        batches = [self.generate_batch() for _ in feature_names]
        return {
            "keys": feature_names,
            "values": torch.cat([b["values"] for b in batches]),
            "lengths": torch.cat([b["lengths"] for b in batches])
        }