#!/usr/bin/env python3
"""Detailed memory profiling to identify leak sources.

This script tracks:
- Tensor allocations by size and device
- Python object types and counts
- Module-level memory usage
- Stack traces for large allocations
"""

import gc
import os
import sys
import time
import psutil
import torch
import tracemalloc
from collections import Counter, defaultdict
from typing import Dict, Any, Optional

class DetailedMemoryProfiler:
    """Advanced memory profiler for identifying leak sources."""
    
    def __init__(self, pid: Optional[int] = None):
        """Initialize profiler.
        
        Args:
            pid: Process ID to monitor (None = this process)
        """
        if pid is None:
            self.process = psutil.Process(os.getpid())
            self.can_introspect = True
        else:
            try:
                self.process = psutil.Process(pid)
                self.can_introspect = False  # Can't introspect external process
                print(f"✓ Monitoring external process {pid}: {self.process.name()}")
            except psutil.NoSuchProcess:
                print(f"❌ Process {pid} not found!")
                sys.exit(1)
        
        self.baseline = None
        self.iteration = 0
        
        # Enable tracemalloc for detailed tracking (only for self)
        if self.can_introspect:
            tracemalloc.start()
            print("✓ Memory tracing enabled")
    
    def analyze_tensors(self) -> Dict[str, Any]:
        """Analyze all tensors in memory."""
        if not self.can_introspect:
            return {}
        
        tensors_by_device = defaultdict(list)
        total_tensor_memory = 0
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    size = obj.element_size() * obj.nelement()
                    device = str(obj.device)
                    tensors_by_device[device].append({
                        'shape': tuple(obj.shape),
                        'dtype': str(obj.dtype),
                        'size_mb': size / 1024 / 1024,
                        'requires_grad': obj.requires_grad
                    })
                    total_tensor_memory += size
            except Exception:
                continue
        
        # Summarize
        summary = {
            'total_tensors': sum(len(v) for v in tensors_by_device.values()),
            'total_memory_mb': total_tensor_memory / 1024 / 1024,
            'by_device': {}
        }
        
        for device, tensors in tensors_by_device.items():
            device_memory = sum(t['size_mb'] for t in tensors)
            # Group by shape
            shape_counts = Counter(t['shape'] for t in tensors)
            
            summary['by_device'][device] = {
                'count': len(tensors),
                'memory_mb': device_memory,
                'unique_shapes': len(shape_counts),
                'top_shapes': shape_counts.most_common(5)
            }
        
        return summary
    
    def analyze_objects(self) -> Dict[str, Any]:
        """Analyze Python objects in memory."""
        if not self.can_introspect:
            return {}
        
        type_counts = Counter()
        type_sizes = defaultdict(int)
        
        for obj in gc.get_objects():
            try:
                obj_type = type(obj).__name__
                type_counts[obj_type] += 1
                try:
                    type_sizes[obj_type] += sys.getsizeof(obj)
                except:
                    pass
            except Exception:
                continue
        
        # Get top consumers
        top_by_count = type_counts.most_common(10)
        top_by_size = sorted(type_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_objects': sum(type_counts.values()),
            'unique_types': len(type_counts),
            'top_by_count': top_by_count,
            'top_by_size': [(t, s/1024/1024) for t, s in top_by_size]
        }
    
    def analyze_memory_blocks(self) -> Dict[str, Any]:
        """Analyze memory allocations using tracemalloc."""
        if not self.can_introspect:
            return {}
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Get top 10 allocation sites
        top_allocations = []
        for stat in top_stats[:10]:
            top_allocations.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        total_size = sum(stat.size for stat in top_stats)
        
        return {
            'total_allocated_mb': total_size / 1024 / 1024,
            'block_count': len(top_stats),
            'top_allocations': top_allocations
        }
    
    def get_process_stats(self) -> Dict[str, Any]:
        """Get process-level statistics."""
        try:
            mem_info = self.process.memory_info()
            
            stats = {
                'rss_mb': mem_info.rss / 1024 / 1024,
                'vms_mb': mem_info.vms / 1024 / 1024,
                'percent': self.process.memory_percent(),
            }
            
            # Try to get more detailed stats
            try:
                mem_full = self.process.memory_full_info()
                stats['uss_mb'] = mem_full.uss / 1024 / 1024  # Unique set size
                stats['pss_mb'] = mem_full.pss / 1024 / 1024  # Proportional set size
            except Exception:
                pass
            
            # Get child processes
            try:
                children = self.process.children(recursive=True)
                stats['child_count'] = len(children)
                stats['child_memory_mb'] = sum(c.memory_info().rss for c in children) / 1024 / 1024
            except Exception:
                stats['child_count'] = 0
                stats['child_memory_mb'] = 0
            
            return stats
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
    
    def print_report(self, label: str = ""):
        """Print comprehensive memory report."""
        self.iteration += 1
        
        print(f"\n{'='*80}")
        print(f"DETAILED MEMORY REPORT - Iteration {self.iteration}{' - ' + label if label else ''}")
        print(f"{'='*80}")
        
        # Process stats
        proc_stats = self.get_process_stats()
        print(f"\n[Process Memory - PID {self.process.pid}]")
        print(f"  RSS (Resident):     {proc_stats.get('rss_mb', 0):.1f} MB")
        print(f"  VMS (Virtual):      {proc_stats.get('vms_mb', 0):.1f} MB")
        if 'uss_mb' in proc_stats:
            print(f"  USS (Unique):       {proc_stats['uss_mb']:.1f} MB")
            print(f"  PSS (Proportional): {proc_stats['pss_mb']:.1f} MB")
        print(f"  Percent:            {proc_stats.get('percent', 0):.1f}%")
        
        if proc_stats.get('child_count', 0) > 0:
            print(f"\n[Child Processes]")
            print(f"  Count:              {proc_stats['child_count']}")
            print(f"  Total Memory:       {proc_stats['child_memory_mb']:.1f} MB")
        
        # System-wide
        system_mem = psutil.virtual_memory()
        print(f"\n[System Memory]")
        print(f"  Used:               {system_mem.used / 1024 / 1024:.1f} MB ({system_mem.percent:.1f}%)")
        print(f"  Available:          {system_mem.available / 1024 / 1024:.1f} MB")
        
        # GPU
        if torch.cuda.is_available():
            print(f"\n[GPU Memory]")
            print(f"  Allocated:          {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
            print(f"  Reserved:           {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB")
            print(f"  Peak:               {torch.cuda.max_memory_allocated() / 1024 / 1024:.1f} MB")
        
        # Detailed analysis (only for self-monitoring)
        if self.can_introspect:
            # Tensors
            tensor_info = self.analyze_tensors()
            if tensor_info:
                print(f"\n[Tensor Analysis]")
                print(f"  Total Tensors:      {tensor_info['total_tensors']}")
                print(f"  Total Memory:       {tensor_info['total_memory_mb']:.1f} MB")
                
                for device, info in tensor_info['by_device'].items():
                    print(f"\n  Device: {device}")
                    print(f"    Count:            {info['count']}")
                    print(f"    Memory:           {info['memory_mb']:.1f} MB")
                    print(f"    Unique Shapes:    {info['unique_shapes']}")
                    if info['top_shapes']:
                        print(f"    Top Shapes:")
                        for shape, count in info['top_shapes'][:3]:
                            print(f"      {str(shape):30s} × {count}")
            
            # Objects
            obj_info = self.analyze_objects()
            if obj_info:
                print(f"\n[Python Objects]")
                print(f"  Total Objects:      {obj_info['total_objects']:,}")
                print(f"  Unique Types:       {obj_info['unique_types']}")
                
                print(f"\n  Top by Count:")
                for obj_type, count in obj_info['top_by_count'][:5]:
                    print(f"    {obj_type:30s} {count:,}")
                
                print(f"\n  Top by Size:")
                for obj_type, size_mb in obj_info['top_by_size'][:5]:
                    print(f"    {obj_type:30s} {size_mb:.1f} MB")
            
            # Memory blocks
            block_info = self.analyze_memory_blocks()
            if block_info:
                print(f"\n[Memory Allocations]")
                print(f"  Total Tracked:      {block_info['total_allocated_mb']:.1f} MB")
                print(f"  Block Count:        {block_info['block_count']:,}")
                
                if block_info['top_allocations']:
                    print(f"\n  Top Allocation Sites:")
                    for alloc in block_info['top_allocations'][:5]:
                        # Clean up the file path
                        file_info = alloc['file'].split('\n')[0] if '\n' in alloc['file'] else alloc['file']
                        print(f"    {alloc['size_mb']:.1f} MB ({alloc['count']:,} blocks)")
                        print(f"      {file_info}")
        
        # Growth analysis
        if self.baseline:
            print(f"\n[Growth from Baseline]")
            
            rss_growth = proc_stats.get('rss_mb', 0) - self.baseline['rss_mb']
            rss_pct = (rss_growth / self.baseline['rss_mb'] * 100) if self.baseline['rss_mb'] > 0 else 0
            print(f"  RSS:                {rss_growth:+.1f} MB ({rss_pct:+.1f}%)")
            
            system_growth = (system_mem.used / 1024 / 1024) - self.baseline['system_mb']
            system_pct = (system_growth / self.baseline['system_mb'] * 100) if self.baseline['system_mb'] > 0 else 0
            print(f"  System:             {system_growth:+.1f} MB ({system_pct:+.1f}%)")
            
            if self.can_introspect and 'total_objects' in self.baseline:
                obj_growth = obj_info['total_objects'] - self.baseline['total_objects']
                obj_pct = (obj_growth / self.baseline['total_objects'] * 100) if self.baseline['total_objects'] > 0 else 0
                print(f"  Python Objects:     {obj_growth:+,} ({obj_pct:+.1f}%)")
                
                if 'total_tensors' in self.baseline:
                    tensor_growth = tensor_info['total_tensors'] - self.baseline['total_tensors']
                    print(f"  Tensors:            {tensor_growth:+,}")
            
            # Flag significant growth
            if rss_growth > 200:
                print(f"\n  ⚠️  WARNING: RSS grew by {rss_growth:.1f} MB!")
            if self.can_introspect and obj_info['total_objects'] - self.baseline.get('total_objects', 0) > 10000:
                obj_growth = obj_info['total_objects'] - self.baseline['total_objects']
                print(f"  ⚠️  WARNING: {obj_growth:,} new Python objects created!")
        
        print(f"{'='*80}\n")
        
        # Set baseline if not set
        if not self.baseline:
            self.baseline = {
                'rss_mb': proc_stats.get('rss_mb', 0),
                'system_mb': system_mem.used / 1024 / 1024,
            }
            if self.can_introspect:
                self.baseline['total_objects'] = obj_info.get('total_objects', 0)
                self.baseline['total_tensors'] = tensor_info.get('total_tensors', 0)
            print("✓ Baseline captured\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Detailed memory profiling')
    parser.add_argument('--pid', type=int, help='Process ID to monitor')
    parser.add_argument('--interval', type=int, default=15, help='Monitoring interval in seconds')
    args = parser.parse_args()
    
    profiler = DetailedMemoryProfiler(pid=args.pid)
    
    if args.pid:
        print(f"Detailed Memory Profiler - Monitoring PID {args.pid}")
    else:
        print("Detailed Memory Profiler - Self-monitoring mode")
    print(f"Update interval: {args.interval} seconds")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            profiler.print_report()
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nProfiling stopped.")
        profiler.print_report("FINAL")
