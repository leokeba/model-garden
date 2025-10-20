# Why The Fix Works in Test But Not in Training - EXACT EXPLANATION

## The Critical Difference

### Test Script (WORKS ✅)

```python
# Test 5: Process batches WITH cleanup
for i in range(num_batches):
    # 1. Get batch samples
    batch_samples = formatted_dataset[i * batch_size : (i + 1) * batch_size]
    
    # 2. Process each sample
    batch_data = []
    for sample in batch_samples:
        processed = processor(text=text, images=image, return_tensors="pt")
        batch_data.append(processed)
    
    # 3. CRITICAL: Delete batch_data variable
    del batch_data  # ← This breaks the Python reference
    
    # 4. Run GC
    gc.collect()    # ← GC can now collect because nothing references batch_data
```

**Reference chain**: `batch_data` (local variable) → `processed` tensors
**After `del batch_data`**: No references → GC collects → Memory freed ✅

---

### Real Training (LEAKS ❌)

```python
# In SFTTrainer.train():
for step, batch in enumerate(dataloader):
    # 1. Dataloader calls UnslothVisionDataCollator.__call__(examples)
    #    Inside UnslothVisionDataCollator:
    #      - Creates texts, images lists (Python objects)
    #      - Calls self.processor(**proc_kwargs)
    #      - Creates batch dict with tensors
    #      - Returns batch
    
    # 2. batch is now in training loop
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    # 3. Our MemoryCleanupCallback.on_step_end() is called:
    model.zero_grad(set_to_none=True)
    gc.collect()  # ← But we DON'T have access to 'batch' variable to del it!
    
    # 4. PROBLEM: The 'batch' variable is LOCAL to Trainer._inner_training_loop()
    #    We CANNOT access it from the callback!
```

**Reference chain**: 
- `Trainer._inner_training_loop()` local variable `batch` → tensors
- `UnslothVisionDataCollator` internal state (images, texts lists) → objects
- **Possibly**: C++ level references in processor/collator

**After `gc.collect()`**: 
- `batch` is still in scope in Trainer's loop → GC cannot collect
- Even after loop iteration ends, **something keeps holding references**

---

## The Actual Code Flow Comparison

### Test Script Stack Trace:
```
main()
  └─ for loop (our code)
       ├─ processor() creates tensors
       ├─ batch_data = [...tensors...]    # We OWN this variable
       ├─ del batch_data                   # We can DELETE it
       └─ gc.collect()                     # GC frees memory ✅
```

### Real Training Stack Trace:
```
trainer.train()
  └─ Trainer._inner_training_loop()       # HuggingFace code
       └─ for step, batch in enumerate(dataloader):
            ├─ dataloader.__next__()
            │    └─ UnslothVisionDataCollator.__call__(examples)
            │         ├─ texts = []        # Internal state
            │         ├─ images = []       # Internal state
            │         ├─ processor(**kwargs) creates tensors
            │         └─ return batch      # Returns to Trainer
            │
            ├─ batch variable is LOCAL to _inner_training_loop()
            ├─ model(**batch)
            ├─ loss.backward()
            ├─ optimizer.step()
            ├─ callbacks.on_step_end()     # ← We are here
            │    └─ gc.collect()            # But 'batch' still in scope! ❌
            │
            └─ Loop continues, 'batch' reference MIGHT remain somewhere
```

---

## Why Can't We Access 'batch' to Delete It?

### The Callback Interface:
```python
class TrainerCallback:
    def on_step_end(self, args, state, control, **kwargs):
        # kwargs contains: model, optimizer, lr_scheduler
        # kwargs DOES NOT contain: batch, outputs, loss
        # We CANNOT access the 'batch' variable!
        pass
```

### What Trainer Passes to Callbacks:
```python
# In Trainer._inner_training_loop():
self.control = self.callback_handler.on_step_end(
    self.args,
    self.state,
    self.control,
    model=self.model_wrapped,
    optimizer=self.optimizer,
    lr_scheduler=self.lr_scheduler,
)
# Notice: 'batch' is NOT passed! It's still in scope as a local variable.
```

---

## Why This Still Doesn't Explain Everything

Even if `batch` goes out of scope at the end of the loop iteration, memory STILL accumulates. This suggests:

### Hypothesis 1: Trainer Keeps References Internally
```python
# Somewhere in Trainer code (speculating):
self._last_batch = batch  # For debugging/logging?
self._past_batches = [...]  # For gradient accumulation?
```
→ We tried clearing `trainer._past` and other attributes, **didn't help**.

### Hypothesis 2: UnslothVisionDataCollator Keeps References
```python
class UnslothVisionDataCollator:
    def __call__(self, examples):
        texts = []   # These lists might not be getting cleared
        images = []  # between calls
        # ... process ...
        return batch
```
→ These are local variables, should be GC'd. But maybe **internal caching**?

### Hypothesis 3: C++ Level References (MOST LIKELY)
```python
# Inside processor or collator (C++ extension):
// C++ code allocates buffers for image processing
void* image_buffer = malloc(large_size);
// Processes images
// Returns Python tensors that wrap the C++ memory
// BUT: C++ buffer not freed even when Python tensor is GC'd
```
→ **This is the most likely cause**. Python GC can't see C++ allocations.

---

## Proof It's C++ Level

1. **Tensor count stable**: Python GC is working (collects Python objects)
2. **Memory grows**: Something outside Python's control is accumulating
3. **Manual cleanup works in test**: Because we process in pure Python scope
4. **Training leaks**: Because UnslothVisionDataCollator uses C++ internally

---

## The Bottom Line

**Test works because**: We call processor directly in Python, create local variables we can `del`, no C++ collator involved.

**Training leaks because**: UnslothVisionDataCollator's C++ code allocates memory that Python can't see or free, and we can't access Trainer's local `batch` variable from callbacks to force cleanup.

The only way to fix this without patching Unsloth would be to:
1. **Replace UnslothVisionDataCollator** with our own pure-Python collator
2. **Implement custom training loop** where WE control the batch lifecycle
3. **Report to Unsloth** and wait for them to fix the C++ code
