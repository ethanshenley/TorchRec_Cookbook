import time
from dataclasses import dataclass
from typing import List, Dict, Callable

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    batch_time_ms: float
    memory_used_gb: float
    throughput: float

class TorchRecBenchmark:
    """Benchmarking utilities for TorchRec"""
    
    def __init__(self, warmup_steps: int = 3, measure_steps: int = 10):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        
    def benchmark_forward(
        self,
        model: torch.nn.Module,
        sample_input: Any,
        batch_size: int
    ) -> BenchmarkResult:
        """Benchmark forward pass"""
        # Warmup
        for _ in range(self.warmup_steps):
            _ = model(sample_input)
            
        # Measure
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        for _ in range(self.measure_steps):
            _ = model(sample_input)
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) * 1000 / self.measure_steps
        memory_used = torch.cuda.max_memory_allocated() / 1e9
        throughput = batch_size * 1000 / avg_time
        
        return BenchmarkResult(avg_time, memory_used, throughput)