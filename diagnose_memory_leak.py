#!/usr/bin/env python3
"""Diagnostic script to monitor memory usage during training.

This script helps identify what's accumulating in memory by tracking:
- System RAM usage (total system or specific process)
- GPU VRAM usage  
- Python object counts
- Tensor allocations
- Garbage collector stats

Usage:
    # Monitor entire system
    python diagnose_memory_leak.py
    
    # Monitor specific process (e.g., training)
    python diagnose_memory_leak.py --pid <process_id>
    
    # Find and monitor training process automatically
    python diagnose_memory_leak.py --find-training
"""

import gc
import os
import sys
import time
import psutil
import torch
import argparse
from collections import Counter
from typing import Dict, Any, Optional

class MemoryMonitor:
    """Monitor and report memory usage."""
    
    def __init__(self, pid: Optional[int] = None):
        """Initialize monitor.
        
        Args:
            pid: Process ID to monitor (None = monitor this process + system total)
        """
        if pid is None:
            self.process = psutil.Process(os.getpid())
            self.monitor_mode = "self"
        else:
            try:
                self.process = psutil.Process(pid)
                self.monitor_mode = "external"
                print(f"✓ Monitoring process {pid}: {self.process.name()}")
                print(f"  Command: {' '.join(self.process.cmdline()[:5])}")
            except psutil.NoSuchProcess:
                print(f"❌ Process {pid} not found!")
                sys.exit(1)
        
        self.baseline_ram = None
        self.baseline_system_ram = None
        self.baseline_gpu = None
        self.step_count = 0
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        stats = {}
        
        # System-wide RAM
        system_mem = psutil.virtual_memory()
        stats['system_ram_mb'] = system_mem.used / 1024 / 1024
        stats['system_ram_percent'] = system_mem.percent
        stats['system_ram_available_mb'] = system_mem.available / 1024 / 1024
        
        # Process RAM
        try:
            mem_info = self.process.memory_info()
            stats['process_ram_mb'] = mem_info.rss / 1024 / 1024
            stats['process_ram_percent'] = self.process.memory_percent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            stats['process_ram_mb'] = 0
            stats['process_ram_percent'] = 0
        
        # GPU VRAM
        if torch.cuda.is_available():
            stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            stats['gpu_peak_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            stats['gpu_allocated_mb'] = 0
            stats['gpu_reserved_mb'] = 0
            stats['gpu_peak_mb'] = 0
        
        # Python objects (only for self-monitoring)
        if self.monitor_mode == "self":
            gc_stats = gc.get_stats()
            stats['gc_collections'] = sum(s['collections'] for s in gc_stats)
            stats['gc_objects'] = len(gc.get_objects())
            
            # Tensor count
            tensor_count = 0
            tensor_memory = 0
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    tensor_count += 1
                    try:
                        tensor_memory += obj.element_size() * obj.nelement()
                    except:
                        pass
            stats['tensor_count'] = tensor_count
            stats['tensor_memory_mb'] = tensor_memory / 1024 / 1024
        else:
            stats['gc_collections'] = 0
            stats['gc_objects'] = 0
            stats['tensor_count'] = 0
            stats['tensor_memory_mb'] = 0
        
        return stats
    
    def print_stats(self, label: str = ""):
        """Print current memory statistics."""
        stats = self.get_memory_stats()
        
        print(f"\n{'='*70}")
        print(f"Memory Stats{' - ' + label if label else ''}")
        print(f"{'='*70}")
        
        # System-wide stats
        print(f"[System-Wide Memory]")
        print(f"  Used:       {stats['system_ram_mb']:.1f} MB ({stats['system_ram_percent']:.1f}%)")
        print(f"  Available:  {stats['system_ram_available_mb']:.1f} MB")
        
        # Process stats
        print(f"\n[Process Memory - PID {self.process.pid}]")
        print(f"  RAM:        {stats['process_ram_mb']:.1f} MB ({stats['process_ram_percent']:.1f}%)")
        
        # GPU stats
        print(f"\n[GPU Memory]")
        print(f"  Allocated:  {stats['gpu_allocated_mb']:.1f} MB")
        print(f"  Reserved:   {stats['gpu_reserved_mb']:.1f} MB")
        print(f"  Peak:       {stats['gpu_peak_mb']:.1f} MB")
        
        # Python internals (only for self-monitoring)
        if self.monitor_mode == "self":
            print(f"\n[Python Internals]")
            print(f"  Objects:    {stats['gc_objects']:,}")
            print(f"  Tensors:    {stats['tensor_count']:,} ({stats['tensor_memory_mb']:.1f} MB)")
            print(f"  GC Runs:    {stats['gc_collections']}")
        
        # Growth from baseline
        if self.baseline_ram is not None:
            print(f"\n[Growth from Baseline]")
            
            # System RAM growth
            system_growth = stats['system_ram_mb'] - self.baseline_system_ram
            print(f"  System RAM:  {system_growth:+.1f} MB ({system_growth/self.baseline_system_ram*100:+.1f}%)")
            
            # Process RAM growth
            process_growth = stats['process_ram_mb'] - self.baseline_ram
            if stats['process_ram_mb'] > 0:
                print(f"  Process RAM: {process_growth:+.1f} MB ({process_growth/self.baseline_ram*100:+.1f}%)")
            
            # GPU growth
            if self.baseline_gpu is not None and self.baseline_gpu > 0:
                gpu_growth = stats['gpu_allocated_mb'] - self.baseline_gpu
                gpu_growth_pct = gpu_growth/self.baseline_gpu*100
                print(f"  GPU:         {gpu_growth:+.1f} MB ({gpu_growth_pct:+.1f}%)")
            else:
                gpu_growth = stats['gpu_allocated_mb'] - (self.baseline_gpu or 0)
                print(f"  GPU:         {gpu_growth:+.1f} MB")
            
            # Flag concerning growth
            if system_growth > 500:
                print(f"\n  ⚠️  WARNING: System RAM grew by {system_growth:.1f} MB!")
            if process_growth > 200:
                print(f"  ⚠️  WARNING: Process RAM grew by {process_growth:.1f} MB!")
        
        print(f"{'='*70}\n")
        
        return stats
    
    def set_baseline(self):
        """Set current state as baseline for growth calculations."""
        stats = self.get_memory_stats()
        self.baseline_ram = stats['process_ram_mb']
        self.baseline_system_ram = stats['system_ram_mb']
        self.baseline_gpu = stats['gpu_allocated_mb']
        print(f"✓ Baseline set:")
        print(f"  System RAM:  {self.baseline_system_ram:.1f} MB")
        print(f"  Process RAM: {self.baseline_ram:.1f} MB")
        print(f"  GPU:         {self.baseline_gpu:.1f} MB")
    
    def analyze_objects(self, top_n: int = 10):
        """Analyze what types of objects are consuming memory."""
        print(f"\n{'='*60}")
        print(f"Top {top_n} Object Types by Count")
        print(f"{'='*60}")
        
        type_counts = Counter()
        for obj in gc.get_objects():
            type_counts[type(obj).__name__] += 1
        
        for obj_type, count in type_counts.most_common(top_n):
            print(f"{obj_type:30s}: {count:,}")
        
        print(f"{'='*60}\n")
    
    def find_tensors(self, min_size_mb: float = 1.0):
        """Find large tensors in memory."""
        print(f"\n{'='*60}")
        print(f"Tensors > {min_size_mb} MB")
        print(f"{'='*60}")
        
        large_tensors = []
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                try:
                    size_mb = obj.element_size() * obj.nelement() / 1024 / 1024
                    if size_mb >= min_size_mb:
                        large_tensors.append({
                            'size_mb': size_mb,
                            'shape': tuple(obj.shape),
                            'dtype': obj.dtype,
                            'device': obj.device,
                            'requires_grad': obj.requires_grad,
                        })
                except:
                    pass
        
        large_tensors.sort(key=lambda x: x['size_mb'], reverse=True)
        
        for i, tensor_info in enumerate(large_tensors[:20], 1):
            print(f"{i:2d}. {tensor_info['size_mb']:.1f} MB - "
                  f"shape={tensor_info['shape']}, "
                  f"dtype={tensor_info['dtype']}, "
                  f"device={tensor_info['device']}, "
                  f"grad={tensor_info['requires_grad']}")
        
        print(f"\nTotal: {len(large_tensors)} tensors > {min_size_mb} MB")
        print(f"{'='*60}\n")


def create_memory_logging_callback():
    """Create a callback for logging memory during training."""
    from transformers import TrainerCallback
    
    class MemoryLoggingCallback(TrainerCallback):
        def __init__(self):
            self.monitor = MemoryMonitor()
            self.step_stats = []
        
        def on_train_begin(self, args, state, control, **kwargs):
            print("\n" + "="*60)
            print("MEMORY MONITORING STARTED")
            print("="*60)
            self.monitor.print_stats("Training Begin")
            self.monitor.set_baseline()
        
        def on_step_end(self, args, state, control, **kwargs):
            # Log every 10 steps
            if state.global_step % 10 == 0:
                stats = self.monitor.get_memory_stats()
                self.step_stats.append({
                    'step': state.global_step,
                    'ram_mb': stats['ram_mb'],
                    'gpu_mb': stats['gpu_allocated_mb'],
                })
                
                # Print summary
                print(f"Step {state.global_step:4d}: "
                      f"RAM={stats['ram_mb']:7.1f}MB "
                      f"GPU={stats['gpu_allocated_mb']:7.1f}MB "
                      f"Tensors={stats['tensor_count']:5d}")
        
        def on_evaluate(self, args, state, control, **kwargs):
            self.monitor.print_stats(f"After Evaluation (step {state.global_step})")
        
        def on_train_end(self, args, state, control, **kwargs):
            self.monitor.print_stats("Training Complete")
            self.monitor.analyze_objects(top_n=15)
            self.monitor.find_tensors(min_size_mb=1.0)
            
            # Print growth summary
            if len(self.step_stats) > 1:
                print("\n" + "="*60)
                print("MEMORY GROWTH ANALYSIS")
                print("="*60)
                first = self.step_stats[0]
                last = self.step_stats[-1]
                ram_growth = last['ram_mb'] - first['ram_mb']
                gpu_growth = last['gpu_mb'] - first['gpu_mb']
                
                print(f"Steps: {first['step']} -> {last['step']} ({last['step'] - first['step']} steps)")
                print(f"RAM:   {first['ram_mb']:.1f} MB -> {last['ram_mb']:.1f} MB ({ram_growth:+.1f} MB)")
                print(f"GPU:   {first['gpu_mb']:.1f} MB -> {last['gpu_mb']:.1f} MB ({gpu_growth:+.1f} MB)")
                
                if ram_growth > 100:
                    print(f"\n⚠️  WARNING: RAM grew by {ram_growth:.1f} MB during training!")
                    print("    This indicates a memory leak.")
                
                print("="*60 + "\n")
    
    return MemoryLoggingCallback()




def find_training_process() -> Optional[int]:
    """Find the model-garden training process."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('train-vision' in arg or 'train' in arg and 'model-garden' in ' '.join(cmdline) for arg in cmdline):
                return proc.info['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Monitor memory usage')
    parser.add_argument('--pid', type=int, help='Process ID to monitor')
    parser.add_argument('--find-training', action='store_true', help='Auto-find training process')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds')
    args = parser.parse_args()
    
    # Determine which process to monitor
    target_pid = None
    if args.find_training:
        target_pid = find_training_process()
        if target_pid is None:
            print("❌ No training process found!")
            print("   Make sure training is running first, then:")
            print("   python diagnose_memory_leak.py --find-training")
            sys.exit(1)
    elif args.pid:
        target_pid = args.pid
    
    # Standalone monitoring
    monitor = MemoryMonitor(pid=target_pid)
    
    if target_pid is None:
        print("Memory Monitor - Monitoring this process + system total")
    else:
        print(f"Memory Monitor - Monitoring external process {target_pid}")
    print(f"Update interval: {args.interval} seconds")
    print("Press Ctrl+C to exit\n")
    
    monitor.set_baseline()
    
    try:
        iteration = 0
        while True:
            time.sleep(args.interval)
            iteration += 1
            
            monitor.print_stats(f"Iteration {iteration}")
            
            # Check for significant growth
            stats = monitor.get_memory_stats()
            system_growth = stats['system_ram_mb'] - monitor.baseline_system_ram
            
            if system_growth > 500 and iteration % 3 == 0:
                print(f"\n⚠️  Significant memory growth detected! Analyzing...")
                if monitor.monitor_mode == "self":
                    monitor.analyze_objects()
                    monitor.find_tensors()
                
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        monitor.print_stats("Final")
        
        # Final analysis
        stats = monitor.get_memory_stats()
        system_growth = stats['system_ram_mb'] - monitor.baseline_system_ram
        process_growth = stats['process_ram_mb'] - monitor.baseline_ram
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Total system RAM growth: {system_growth:+.1f} MB")
        print(f"Total process RAM growth: {process_growth:+.1f} MB")
        
        if system_growth > 200:
            print(f"\n⚠️  System RAM grew significantly!")
            print("   This indicates a memory leak in the training process.")
        elif process_growth > 100:
            print(f"\n⚠️  Process RAM grew significantly!")
            print("   Check if this is expected for your workload.")
        else:
            print(f"\n✓ Memory usage looks stable.")
        print("="*70)

