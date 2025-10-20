# Memory Leak Investigation - Findings

## Executive Summary

Initial hypothesis about `torch.compile()` workers was **PARTIALLY CORRECT** but **INCOMPLETE**. The real situation is more nuanced.

## Key Findings

### 1. Multiple Memory Issues Identified

| Issue | Impact | Status |
|-------|--------|--------|
| Compile workers overhead | 7-8 GB constant | ⚠️ Not a leak, just overhead |
| Validation spikes | +4 GB temporary | ✅ Normal, GC recovers |
| Slow baseline leak | TBD | 🔍 Currently testing |

### 2. What We Learned

#### The Tool Was Monitoring the Wrong Process

Your observation was crucial:
```
System RAM: +1112 MB  ⚠️ Growing
Process RAM: +3.5 MB  ✅ Stable (PID 142083 - just the wrapper)
```

This revealed:
- **PID 142083**: `uv run model-garden` wrapper - stable
- **PID 141495**: Actual training process - the one leaking
- **20 child processes**: compile_worker processes - stable but using 8 GB

#### Compile Workers Are NOT Leaking

The compile workers:
- ✅ Stable at ~400 MB each
- ✅ Don't grow over time
- ⚠️ But use 7-8 GB total (overhead)
- ⚠️ Only exist during training

#### Validation Causes Huge Spikes

Observed behavior:
```
13:18:20  6.2 GB  <- Training
13:18:25  10.4 GB  <- Validation starts (+4 GB spike!)
13:18:30  4.8 GB  <- GC recovered
```

This is likely **normal** - validation loads all images at once.

### 3. The Real Question

**Is there a SLOW LEAK underneath the spikes?**

To answer this, we need to track the **minimum memory** over time:
- If minimum rises: 🔴 SLOW LEAK
- If minimum stable: 🟢 JUST SPIKES (normal)

Currently running: `track_leak_precisely.py` to measure this.

## Technical Details

### Process Tree

```
PID 141489  - uv run model-garden serve (wrapper)
  └─ PID 141495 - model-garden serve (main process) ← THE ONE TO WATCH
       ├─ PID 141614 - compile_worker (coordinator) - 665 MB
       ├─ PID 141658 - compile_worker - 398 MB
       ├─ PID 141660 - compile_worker - 397 MB
       └─ ... (18 more workers) ... ~400 MB each
```

### Memory Accounting

Total system memory used by training:
- Main process: ~6.2 GB (variable, spikes to 10 GB)
- Compile workers: ~8 GB total (stable)
- **Total overhead: ~14 GB minimum**

### Compile Worker Behavior

The compile workers are created by PyTorch's Inductor for JIT compilation:
- Created on first training iteration
- Persist for the entire training session
- Destroyed when training ends
- Use shared memory with parent process (so not full 8 GB overhead)

## Hypotheses to Test

### A. No Leak - Just Spikes (POSSIBLE)

If the leak detector shows:
```
First 10 samples minimum: 6000 MB
Last 10 samples minimum:  6050 MB  (+50 MB)
Result: 🟢 NO LEAK
```

Then the "leak" is just **validation spikes + GC lag**, which is normal.

**Action**: Increase `cleanup_every_n_steps` to reduce GC overhead.

### B. Slow Leak Exists (POSSIBLE)

If the leak detector shows:
```
First 10 samples minimum: 6000 MB
Last 10 samples minimum:  7000 MB  (+1000 MB)
Result: 🔴 LEAK DETECTED
```

Then something is accumulating despite GC.

**Possible causes**:
1. **Image caching** - PIL images not being freed
2. **Gradient accumulation** - Old gradients lingering
3. **Callback state** - Metrics/logs accumulating
4. **Tensor refs** - Tensors kept in closures/caches
5. **Compile cache** - Inductor caching growing

**Action**: Use `profile_memory_detailed.py` to identify what's accumulating.

### C. Compile Workers ARE Leaking (UNLIKELY)

If individual workers grow over time, that's a PyTorch bug.

**Action**: Disable torch.compile() as originally planned.

## Next Steps

1. ⏳ **Wait for leak tracker to complete** (5 minutes)
2. 🔍 **Analyze results** - leak or no leak?
3. If leak detected:
   - Run `profile_memory_detailed.py --pid 141495` for object-level analysis
   - Identify what's accumulating (tensors? objects? memory blocks?)
   - Apply targeted fix

4. If no leak:
   - Consider the 14 GB overhead acceptable (compile workers are worth it)
   - Or disable torch.compile() to save 8 GB if memory is tight
   - Document that spikes are normal validation behavior

## Tools Created

### 1. `diagnose_memory_leak.py`
- Monitors process + system RAM
- Shows GPU memory
- Tracks Python objects (self-monitoring only)
- ✅ Works but doesn't separate spikes from leaks

### 2. `profile_memory_detailed.py`
- Deep object-level analysis
- Tensor tracking by device/shape
- Memory allocation tracing
- 🎯 Use this to identify WHAT is leaking

### 3. `track_leak_precisely.py`
- Tracks min/max/average over time
- Separates GC spikes from true leaks
- Compares first vs last minimum
- 🎯 Use this to detect IF there's a leak

## Current Status

🔍 **Running leak detection** - will know in ~3 more minutes if there's a true leak.

Preliminary assessment:
- ✅ Compile workers: overhead but not leaking
- ✅ Validation spikes: normal behavior
- ❓ Slow baseline leak: testing now...

## Recommendations (Preliminary)

### If No Leak Found:
1. Accept 14 GB memory overhead as cost of optimization
2. Ensure enough system RAM (32 GB+ recommended)
3. Monitor minimum memory stays stable
4. Document validation spikes as normal

### If Leak Found:
1. Use detailed profiler to identify source
2. Apply targeted fix (likely in vision_training.py callbacks)
3. Re-test with leak tracker
4. Consider torch.compile() disable as last resort

## Lessons Learned

1. ✅ **Monitor the right process** - wrappers != actual process
2. ✅ **Track children too** - memory can hide in subprocesses
3. ✅ **Separate spikes from leaks** - need min/max/avg tracking
4. ⚠️ **Don't jump to conclusions** - torch.compile() was suspected but may not be the culprit
5. 🎯 **Measure precisely** - "memory is growing" needs quantification

---

**Status**: Investigation ongoing  
**Next Update**: After leak tracker completes (~3 minutes)
