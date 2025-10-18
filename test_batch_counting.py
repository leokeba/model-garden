"""Test to verify batch counting and auto-detection logic."""

# Mock the collator's behavior
class MockCollator:
    def __init__(self):
        self.batch_count = 0
        self.detection_batches = 10
        self.auto_detect = True
        self.auto_detection_complete = False
        self.schema_keys = set()
        self.detected_keys_counter = {}
        self.mask_keys = True
        
    def process_batch(self, batch_num):
        """Simulate processing one batch."""
        print(f"\n=== Batch {batch_num} ===")
        print(f"batch_count before: {self.batch_count}")
        
        # Auto-detect schema keys if needed
        if self.auto_detect and not self.auto_detection_complete:
            if self.batch_count < self.detection_batches:
                print("  → Detecting schema keys...")
                self._detect_schema_keys_from_batch()
            else:
                print("  → Finalizing schema key detection...")
                self._finalize_schema_key_detection()
        
        # Apply masking
        print(f"  → Applying masking")
        print(f"     mask_keys={self.mask_keys}, schema_keys={self.schema_keys}")
        if self.mask_keys and self.schema_keys:
            print(f"     ✅ Schema key masking ACTIVE ({len(self.schema_keys)} keys)")
        else:
            print(f"     ⚠️  Schema key masking INACTIVE")
        
        # Increment batch count
        self.batch_count += 1
        print(f"batch_count after: {self.batch_count}")
        
    def _detect_schema_keys_from_batch(self):
        """Mock detection - just increment counter."""
        # In real code, this extracts keys from JSON
        # For simulation, just say we found some keys
        if self.batch_count == 0:
            self.detected_keys_counter = {"field1": 1, "field2": 1}
        else:
            self.detected_keys_counter["field1"] = self.detected_keys_counter.get("field1", 0) + 1
            self.detected_keys_counter["field2"] = self.detected_keys_counter.get("field2", 0) + 1
    
    def _finalize_schema_key_detection(self):
        """Mock finalization - select frequent keys."""
        if self.auto_detection_complete:
            return
        
        threshold = self.detection_batches * 0.3
        detected_keys = {
            key for key, count in self.detected_keys_counter.items()
            if count >= threshold
        }
        
        self.schema_keys = detected_keys
        self.auto_detection_complete = True
        print(f"  → Finalized! Detected {len(detected_keys)} keys: {detected_keys}")

# Simulate processing batches
collator = MockCollator()

print("=" * 60)
print("SIMULATION: First 15 batches")
print("=" * 60)

for i in range(15):
    collator.process_batch(i)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Schema keys detected: {collator.schema_keys}")
print(f"Auto-detection complete: {collator.auto_detection_complete}")
print(f"Batches where schema masking was ACTIVE: 10+")
print(f"Batches where schema masking was INACTIVE: 0-9")
