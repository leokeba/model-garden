#!/usr/bin/env python3
"""Continuously monitor the FastAPI server process memory during training.

This script monitors the main process and will detect when training starts
(by watching for memory growth patterns) and track the memory usage.
"""

import time
import sys
from datetime import datetime


def get_process_memory_mb(pid):
    """Get process memory in MB."""
    try:
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    kb = int(line.split()[1])
                    return kb / 1024
    except Exception:
        return None
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_training_memory.py <PID>")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    
    print(f"Monitoring PID {pid} - waiting for training to start...")
    print("Press Ctrl+C to stop")
    print("-" * 80)
    
    baseline = None
    training_started = False
    step = 0
    memory_samples = []
    
    try:
        while True:
            mem = get_process_memory_mb(pid)
            
            if mem is None:
                print(f"Process {pid} not found!")
                break
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            step += 1
            
            # Detect training start (significant memory increase)
            if baseline is None:
                baseline = mem
            elif not training_started and mem > baseline + 500:
                training_started = True
                print(f"\n🚀 Training detected! Memory jumped from {baseline:.1f} MB to {mem:.1f} MB")
                print(f"{'Time':<10} {'Step':<6} {'Memory (MB)':<15} {'Change':<15} {'Status'}")
                print("-" * 80)
                memory_samples = [mem]
            
            # Track during training
            if training_started:
                memory_samples.append(mem)
                
                # Calculate change from start of training
                training_start_mem = memory_samples[0]
                change = mem - training_start_mem
                
                # Calculate rate (MB/minute)
                if len(memory_samples) >= 30:  # After 1 minute
                    recent_samples = memory_samples[-30:]
                    rate_per_min = (recent_samples[-1] - recent_samples[0]) * 30  # 30 samples = 1 min
                    status = f"📈 {rate_per_min:+.1f} MB/min" if abs(rate_per_min) > 10 else "✅ STABLE"
                else:
                    status = "Measuring..."
                
                print(f"{timestamp:<10} {step:<6} {mem:>10.1f} MB    {change:>+10.1f} MB    {status}")
            else:
                # Still waiting for training
                if step % 10 == 0:
                    print(f"[{timestamp}] Waiting... (current: {mem:.1f} MB, baseline: {baseline:.1f} MB)")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Monitoring stopped")
        
        if training_started and len(memory_samples) > 1:
            total_change = memory_samples[-1] - memory_samples[0]
            duration_min = len(memory_samples) * 2 / 60  # 2 seconds per sample
            rate = total_change / duration_min if duration_min > 0 else 0
            
            print(f"\nTraining Summary:")
            print(f"  Duration: {duration_min:.1f} minutes")
            print(f"  Start memory: {memory_samples[0]:.1f} MB")
            print(f"  End memory: {memory_samples[-1]:.1f} MB")
            print(f"  Total change: {total_change:+.1f} MB")
            print(f"  Rate: {rate:+.1f} MB/minute")
            
            if abs(rate) < 10:
                print(f"  ✅ LEAK FIXED - Memory is stable!")
            else:
                print(f"  ⚠️  LEAK DETECTED - Memory growing at {rate:.1f} MB/min")


if __name__ == "__main__":
    main()
