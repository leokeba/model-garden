#!/usr/bin/env python3
"""Precise leak tracking - monitors min/max/average to detect slow leaks under GC spikes."""

import time
import psutil
import sys
from collections import deque

def track_memory(pid, duration_minutes=10, sample_interval=2):
    """Track memory with min/max/avg to see past GC spikes.
    
    Args:
        pid: Process ID to monitor
        duration_minutes: How long to monitor
        sample_interval: Seconds between samples
    """
    try:
        process = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"❌ Process {pid} not found!")
        sys.exit(1)
    
    print(f"🔍 Tracking PID {pid}: {process.name()}")
    print(f"Duration: {duration_minutes} minutes, sampling every {sample_interval}s")
    print(f"\nLooking for SLOW LEAK (rising minimum) vs SPIKES (high max)\n")
    print(f"{'Time':>8} {'Current':>10} {'Min':>10} {'Max':>10} {'Avg':>10} {'Trend':>15}")
    print("=" * 70)
    
    samples = deque(maxlen=30)  # Rolling window
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    min_ever = float('inf')
    max_ever = 0
    
    try:
        while time.time() < end_time:
            try:
                rss_mb = process.memory_info().rss / 1024 / 1024
                samples.append(rss_mb)
                
                current_min = min(samples)
                current_max = max(samples)
                current_avg = sum(samples) / len(samples)
                
                min_ever = min(min_ever, rss_mb)
                max_ever = max(max_ever, rss_mb)
                
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                
                # Detect trend
                if len(samples) >= 20:
                    first_half_avg = sum(list(samples)[:10]) / 10
                    second_half_avg = sum(list(samples)[-10:]) / 10
                    diff = second_half_avg - first_half_avg
                    
                    if diff > 50:
                        trend = f"📈 LEAK +{diff:.0f}MB"
                    elif diff < -50:
                        trend = f"📉 DROP {diff:.0f}MB"
                    else:
                        trend = f"➡️  STABLE {diff:+.0f}MB"
                else:
                    trend = "⏳ warming up"
                
                print(f"{mins:02d}:{secs:02d}   {rss_mb:>8.0f} MB {current_min:>8.0f} MB {current_max:>8.0f} MB {current_avg:>8.0f} MB {trend:>15}")
                
                time.sleep(sample_interval)
                
            except psutil.NoSuchProcess:
                print("\n❌ Process ended")
                break
                
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped")
    
    print(f"\n📊 SUMMARY")
    print("=" * 70)
    print(f"Lowest memory seen:  {min_ever:.0f} MB")
    print(f"Highest memory seen: {max_ever:.0f} MB")
    print(f"Range (spike size):  {max_ever - min_ever:.0f} MB")
    
    elapsed_total = int(time.time() - start_time)
    
    if len(samples) >= 20:
        first_10_min = min(list(samples)[:10])
        last_10_min = min(list(samples)[-10:])
        leak_amount = last_10_min - first_10_min
        
        print(f"\n🔍 LEAK DETECTION (comparing minimum values):")
        print(f"First 10 samples minimum: {first_10_min:.0f} MB")
        print(f"Last 10 samples minimum:  {last_10_min:.0f} MB")
        print(f"Change in floor:          {leak_amount:+.0f} MB")
        
        if leak_amount > 100:
            print(f"\n🔴 LEAK DETECTED: Baseline rising by {leak_amount:.0f} MB")
            rate_per_min = leak_amount / (elapsed_total / 60)
            print(f"   Leak rate: ~{rate_per_min:.1f} MB/minute")
        elif leak_amount > 50:
            print(f"\n🟡 POSSIBLE LEAK: Small baseline increase of {leak_amount:.0f} MB")
        else:
            print(f"\n🟢 NO LEAK: Memory floor is stable")
            print(f"   (Spikes of {max_ever - min_ever:.0f} MB are normal GC behavior)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python track_leak_precisely.py <PID> [duration_minutes] [sample_interval]")
        sys.exit(1)
    
    pid = int(sys.argv[1])
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    interval = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    track_memory(pid, duration, interval)
