import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

class TorchRecVisualizer:
    """Visualization utilities for TorchRec"""
    
    @staticmethod
    def plot_memory_usage(memory_stats: List[Dict[str, float]]):
        """Plot memory usage over time"""
        timestamps = list(range(len(memory_stats)))
        allocated = [s["allocated"] / 1e9 for s in memory_stats]
        reserved = [s["reserved"] / 1e9 for s in memory_stats]
        
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, allocated, label="Allocated")
        plt.plot(timestamps, reserved, label="Reserved")
        plt.xlabel("Time")
        plt.ylabel("Memory (GB)")
        plt.title("GPU Memory Usage")
        plt.legend()
        plt.grid(True)
        
    @staticmethod
    def plot_embedding_distribution(embeddings: torch.Tensor):
        """Visualize embedding value distributions"""
        plt.figure(figsize=(10, 5))
        plt.hist(embeddings.detach().cpu().numpy().flatten(), bins=50)
        plt.xlabel("Embedding Values")
        plt.ylabel("Frequency")
        plt.title("Embedding Distribution")
        plt.grid(True)