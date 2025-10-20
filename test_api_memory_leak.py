"""Simplified production-like test using the API.

Instead of loading the model again, we'll trigger training via the API
(which uses the already-loaded model) and monitor memory from outside.
"""

import time
import requests
import psutil

API_BASE = "http://localhost:8000"

def get_service_memory():
    """Get memory usage of the model-garden service."""
    import subprocess
    result = subprocess.run(
        ["systemctl", "status", "model-garden.service", "--no-pager"],
        capture_output=True,
        text=True
    )
    for line in result.stdout.split('\n'):
        if 'Memory:' in line:
            # Extract memory value (e.g., "Memory: 5.1G" -> 5.1)
            mem_str = line.split('Memory:')[1].strip()
            if 'G' in mem_str:
                return float(mem_str.replace('G', '')) * 1024  # Convert to MB
            elif 'M' in mem_str:
                return float(mem_str.replace('M', ''))
    return None

print("=" * 80)
print("API-BASED PRODUCTION MEMORY LEAK TEST")
print("=" * 80)

# Get baseline memory
print("\n1. Getting baseline memory...")
baseline_mem = get_service_memory()
print(f"   Service memory: {baseline_mem:.1f} MB")

# Trigger training via API (small job)
print("\n2. Triggering training job via API...")
print("   Config: 100 samples, 50 max_steps, batch_size=2")

payload = {
    "dataset_name": "Barth371/cmr-all",
    "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "max_steps": 50,  # Only 50 steps
    "dataset_sample_size": 100,  # Only 100 samples
}

try:
    response = requests.post(f"{API_BASE}/train/vision", json=payload, timeout=5)
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data.get("job_id")
        print(f"   ✓ Job started: {job_id}")
    else:
        print(f"   ✗ Failed to start job: {response.status_code}")
        print(f"     {response.text}")
        exit(1)
except requests.exceptions.Timeout:
    print("   ✓ Job started (request timed out, but that's expected)")
    job_id = "unknown"
except Exception as e:
    print(f"   ✗ Error: {e}")
    exit(1)

# Monitor memory every 15 seconds
print("\n3. Monitoring memory during training...")
print("   (Collecting data every 15 seconds)")
print("-" * 80)

memory_history = [(0, baseline_mem)]
start_time = time.time()

try:
    while True:
        time.sleep(15)
        
        elapsed = time.time() - start_time
        current_mem = get_service_memory()
        
        if current_mem is None:
            print("   ✗ Could not get memory info")
            break
        
        memory_history.append((elapsed, current_mem))
        growth = current_mem - baseline_mem
        
        print(f"   [{elapsed/60:.1f} min] Memory: {current_mem:.1f} MB (+{growth:.1f} MB)")
        
        # Stop after 5 minutes or if memory stops growing (training finished)
        if elapsed > 300:
            print("   Stopping after 5 minutes...")
            break
        
        # Check if training is done (memory stabilizes)
        if len(memory_history) > 4:
            recent_growth = memory_history[-1][1] - memory_history[-4][1]
            if abs(recent_growth) < 100:  # Less than 100 MB growth in last minute
                print("   Memory stabilized, training likely finished")
                break

except KeyboardInterrupt:
    print("\n   Interrupted by user")

print("-" * 80)

# Analysis
print("\n4. Analysis:")
print(f"   Duration: {(time.time() - start_time)/60:.1f} minutes")
print(f"   Baseline memory: {baseline_mem:.1f} MB")
print(f"   Final memory: {memory_history[-1][1]:.1f} MB")
print(f"   Total growth: +{memory_history[-1][1] - baseline_mem:.1f} MB")

if len(memory_history) > 2:
    print("\n   Memory timeline:")
    for elapsed, mem in memory_history:
        growth = mem - baseline_mem
        print(f"     {elapsed/60:5.1f} min: {mem:7.1f} MB (+{growth:6.1f} MB)")
    
    # Calculate leak rate
    total_growth = memory_history[-1][1] - memory_history[0][1]
    total_time_min = (memory_history[-1][0] - memory_history[0][0]) / 60
    
    if total_time_min > 0:
        leak_rate_per_min = total_growth / total_time_min
        print(f"\n   Leak rate: {leak_rate_per_min:.1f} MB/minute")
        
        if leak_rate_per_min > 500:
            print("   ⚠️  SEVERE LEAK confirmed (~1 GB/minute)")
        elif leak_rate_per_min > 100:
            print("   ⚠️  Moderate leak detected")
        else:
            print("   ✓ No significant leak (normal memory growth)")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
