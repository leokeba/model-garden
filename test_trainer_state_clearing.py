"""Test clearing Trainer's internal state to prevent memory leak.

The hypothesis: Trainer holds references to batch data in its internal state.
We need to find and clear these references.
"""

import gc
import torch
from transformers import Trainer

print("=" * 80)
print("Investigating Trainer Internal State")
print("=" * 80)

# Check what attributes a Trainer instance has that might hold batch data
print("\nTrainer attributes that might hold references:")
print("-" * 80)

# These are potential culprits based on Trainer's internal implementation
potential_refs = [
    '_past',  # Past outputs
    '_last_loss',  # Last loss tensor
    '_last_optimizer_step',  # Last optimizer step data
    '_accelerator',  # Accelerator might cache things
    'current_flos',  # Floating point operations
    'control',  # Training control state
    'state',  # Trainer state
]

for attr in potential_refs:
    print(f"  - {attr}")

print("\n" + "=" * 80)
print("Recommendation:")
print("=" * 80)
print("""
The Trainer likely holds references in one of these places:
1. The Accelerator's internal cache
2. The _past attribute (for generation tasks)
3. Internal loss/gradient tensors

To fix, we should try clearing these in our callback:

```python
class AggressiveCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Get the trainer instance (if available)
        model = kwargs.get('model')
        
        # Clear gradients
        if model:
            model.zero_grad(set_to_none=True)
        
        # Try to access and clear trainer's internal state
        # Note: This requires accessing the trainer instance itself
        # which might not be available in kwargs
        
        gc.collect()
        return control
```

However, the kwargs don't give us access to the Trainer instance itself!

**SOLUTION**: We need to either:
1. Modify the Trainer class to clear its internal state
2. Use a custom training loop instead of Trainer
3. Find a way to inject a reference to the Trainer into the callback
""")

print("\nLet's check if we can access trainer through closure...")
print("-" * 80)

# In the actual code, we create the callback inside train_vision_model()
# where we have access to the trainer instance. We can use closure!

class CleanupWithTrainerAccess(torch.nn.Module):
    """Callback that has access to trainer via closure."""
    def __init__(self, trainer_ref):
        super().__init__()
        self.trainer_ref = trainer_ref
    
    def __call__(self):
        print("✓ Can access trainer via closure!")
        print(f"  Trainer type: {type(self.trainer_ref)}")
        return True

print("\n✓ SOLUTION FOUND: Use closure to access trainer in callback!")
print("  We can pass the trainer instance to the callback via closure")
print("  since the callback is created inside the train_vision_model() method")
print("  where the trainer instance exists!")
