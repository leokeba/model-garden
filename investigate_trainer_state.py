"""Test if we can access and clear trainer's internal state from a callback."""

import gc
import torch
from transformers import TrainerCallback

# Check what attributes the Trainer has that might hold batch data
print("Investigating HuggingFace Trainer attributes...")
print("=" * 80)

# We'll need to run this during actual training to see what's there
# For now, let's create a test callback that logs all trainer attributes

class InvestigationCallback(TrainerCallback):
    def __init__(self):
        self.logged = False
        
    def on_step_end(self, args, state, control, **kwargs):
        if not self.logged and state.global_step == 1:
            self.logged = True
            print("\n" + "=" * 80)
            print("TRAINER KWARGS AT STEP 1:")
            print("=" * 80)
            for key in sorted(kwargs.keys()):
                value = kwargs[key]
                print(f"{key}: {type(value).__name__}")
                
                # If it's the model, check its attributes
                if key == "model":
                    print(f"  Model attributes:")
                    for attr in dir(value):
                        if not attr.startswith('_') and hasattr(value, attr):
                            attr_value = getattr(value, attr)
                            if torch.is_tensor(attr_value):
                                print(f"    {attr}: Tensor {attr_value.shape}")
                                
            print("=" * 80)
        return control

# Save this callback definition for use in vision_training.py
callback_code = '''
class AggressiveBatchCleanupCallback(TrainerCallback):
    """Callback that explicitly clears batch data to prevent memory leaks."""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Clear all possible references to batch data after each step."""
        
        # 1. Clear model gradients
        model = kwargs.get('model')
        if model is not None and hasattr(model, 'zero_grad'):
            model.zero_grad(set_to_none=True)
        
        # 2. Try to clear any cached data in kwargs
        # The trainer passes various objects that might hold references
        for key in list(kwargs.keys()):
            if key not in ['model', 'args', 'state', 'control']:
                # Don't delete core objects, but clear transient ones
                value = kwargs.get(key)
                if torch.is_tensor(value):
                    del kwargs[key]
        
        # 3. Force garbage collection
        gc.collect()
        
        # 4. Clear CUDA cache (though this is VRAM, not RAM)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return control
'''

print("\nCallback code to try:")
print(callback_code)

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("1. Add InvestigationCallback to training to see what Trainer passes")
print("2. Identify which kwargs hold tensor data")
print("3. Modify AggressiveBatchCleanupCallback to clear those specific items")
print("4. Test if this reduces the memory leak")
