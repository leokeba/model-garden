# Testing Guide for Training Job Rerun Feature

## Prerequisites

1. Model Garden API server running
2. Frontend built and accessible
3. At least one completed, failed, or cancelled training job in the system
4. Access to a GPU for actual training tests (optional - can test UI flow without)

## Test Cases

### 1. Basic Rerun from List Page

**Steps**:
1. Navigate to `/training` page
2. Locate a completed/failed/cancelled job
3. Verify the "🔄 Rerun" button is visible next to the "Delete" button
4. Click the "🔄 Rerun" button
5. Verify confirmation dialog appears with job name
6. Click "OK" in the confirmation dialog
7. Verify button shows "Starting..." loading state
8. Verify redirect to new job details page
9. Verify new job has timestamp suffix in name (e.g., `_rerun_20251022_143025`)
10. Verify new job starts with "queued" status

**Expected Result**: 
- New job created with same configuration
- User redirected to new job page
- Original job remains unchanged in the list

**API Call**:
```bash
curl -X POST http://localhost:8000/api/v1/training/jobs/{original_job_id}/rerun
```

### 2. Basic Rerun from Details Page

**Steps**:
1. Navigate to a completed/failed/cancelled job details page
2. Scroll to "Actions" section
3. Verify "🔄 Rerun Training" button is visible
4. Click the button
5. Verify detailed confirmation dialog
6. Confirm the action
7. Verify redirect to new job page

**Expected Result**: Same as Test Case 1

### 3. Configuration Preservation - Text Model

**Setup**: Use a text-only training job with these settings:
- Base model: `unsloth/llama-3-8b-bnb-4bit`
- Dataset: Custom JSONL
- Validation dataset: Enabled
- Quality mode: Enabled (16-bit)
- Early stopping: Enabled (patience 3)
- LoRA rank: 32

**Steps**:
1. Rerun the job
2. Navigate to new job details
3. Verify Configuration section shows:
   - Quality Mode: ✓ Enabled (16-bit) badge
   - Early Stopping: ✓ Enabled (patience: 3) badge
   - Validation Dataset: ✓ Enabled badge
4. Verify Hyperparameters section matches original
5. Verify LoRA Configuration section shows rank 32

**Expected Result**: All parameters exactly match the original job

### 4. Configuration Preservation - Vision Model

**Setup**: Use a vision-language model training job with:
- Base model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Model type: vision
- Selective loss: Enabled (moderate)
- 8-bit precision

**Steps**:
1. Rerun the job
2. Verify Configuration shows:
   - Model Type: vision
   - Quality Mode: 8-bit Precision badge
   - Selective Loss: ✓ moderate badge
3. Verify vision-specific hyperparameters preserved

**Expected Result**: Vision model settings correctly preserved

### 5. Rerun Metadata Tracking

**Steps**:
1. Create a rerun from job "original-model"
2. View the new job details
3. Verify "Rerun From" field shows "original-model" as a link
4. Click the link
5. Verify navigation to original job

**Expected Result**: Clear bidirectional linkage between jobs

### 6. Multiple Reruns

**Steps**:
1. Rerun job A → creates job B
2. Complete job B (or let it fail)
3. Rerun job B → creates job C
4. Verify job C has "Rerun From: B" 
5. Verify job B has "Rerun From: A"
6. Verify each rerun has unique timestamp suffix

**Expected Result**: Chain of reruns with clear lineage

### 7. Output Directory Isolation

**Steps**:
1. Note original job output directory
2. Rerun the job
3. Verify new job output directory is different
4. Format: `original_dir_parent/job_name_rerun_TIMESTAMP`
5. Verify training writes to new directory (if running actual training)

**Expected Result**: No conflicts between original and rerun model files

### 8. Error Handling - Cannot Rerun Active Job

**Steps**:
1. Start a training job
2. While job is "running" or "queued", attempt to rerun via API:
   ```bash
   curl -X POST http://localhost:8000/api/v1/training/jobs/{running_job_id}/rerun
   ```
3. Verify error response

**Expected Result**:
```json
{
  "success": false,
  "message": "Cannot rerun a job that is currently running. Cancel it first."
}
```

**UI Verification**: The "🔄 Rerun" button should not be visible for running/queued jobs

### 9. Error Handling - Nonexistent Job

**Steps**:
1. Attempt to rerun a job that doesn't exist:
   ```bash
   curl -X POST http://localhost:8000/api/v1/training/jobs/fake-uuid/rerun
   ```

**Expected Result**: 404 error with message "Training job fake-uuid not found"

### 10. Quality Mode Variations

Test all three precision levels:

**Test 10a: 4-bit (Default)**
- Original job has no quality settings
- Verify rerun shows: "4-bit (Default)" in gray text

**Test 10b: 8-bit**
- Original job has `load_in_8bit: true`
- Verify rerun shows: "8-bit Precision" yellow badge

**Test 10c: 16-bit**
- Original job has `load_in_16bit: true` or `quality_mode: true`
- Verify rerun shows: "✓ Enabled (16-bit)" blue badge

### 11. Selective Loss Parameter Preservation

**Setup**: Job with selective loss configuration:
```json
{
  "selective_loss": true,
  "selective_loss_level": "moderate",
  "selective_loss_schema_keys": ["name", "age"],
  "selective_loss_masking_start_epoch": 0.5,
  "selective_loss_verbose": true
}
```

**Steps**:
1. Rerun the job
2. View API response for new job
3. Verify all selective loss parameters preserved

### 12. Early Stopping Variations

**Test 12a**: No early stopping
- Verify rerun doesn't show early stopping badge

**Test 12b**: Early stopping with custom patience
- Original: `early_stopping_patience: 5`
- Verify rerun shows: "✓ Enabled (patience: 5)"

**Test 12c**: Early stopping with threshold
- Verify threshold preserved in rerun job data

### 13. HuggingFace Hub Dataset Flags

**Setup**: Job using HuggingFace datasets:
```json
{
  "dataset_path": "yahma/alpaca-cleaned",
  "from_hub": true,
  "validation_dataset_path": "yahma/alpaca-cleaned",
  "validation_from_hub": true
}
```

**Steps**:
1. Rerun the job
2. Verify both `from_hub` and `validation_from_hub` flags preserved
3. Verify job loads datasets from Hub correctly

### 14. UI Button States

**Test 14a**: Running Job
- Verify shows: "⏸️ Stop Early" and "Cancel" buttons
- Verify NO "🔄 Rerun" button

**Test 14b**: Completed Job
- Verify shows: "🔄 Rerun Training" button
- Verify button is primary/blue color

**Test 14c**: Failed Job
- Verify shows: "🔄 Rerun Training" button
- Encourages retry with same settings

**Test 14d**: Cancelled Job
- Verify shows: "🔄 Rerun Training" button
- Allows restart of cancelled job

### 15. Queue Integration

**Steps**:
1. Start a long-running training job (queue will be busy)
2. Rerun another job while first is running
3. Verify rerun job enters queue
4. Check response includes queue position:
   ```json
   {
     "queue_position": 2,
     "message": "Training job rerun created and queued for execution (position in queue: 2)"
   }
   ```

### 16. All Hyperparameters Preserved

Verify these are preserved:
- `learning_rate`
- `num_epochs`
- `batch_size`
- `gradient_accumulation_steps`
- `max_steps`
- `warmup_steps`
- `logging_steps`
- `save_steps`
- `eval_steps`
- `optim`
- `weight_decay`
- `lr_scheduler_type`
- `max_grad_norm`
- `adam_beta1`, `adam_beta2`, `adam_epsilon`
- `dataloader_num_workers`
- `eval_strategy`
- `load_best_model_at_end`
- `metric_for_best_model`
- `save_total_limit`

### 17. All LoRA Parameters Preserved

Verify these are preserved:
- `r` (rank)
- `lora_alpha`
- `lora_dropout`
- `lora_bias`
- `use_rslora`
- `use_gradient_checkpointing`
- `random_state`
- `target_modules`
- `task_type`
- Vision-specific: `finetune_vision_layers`, `finetune_language_layers`, etc.

### 18. Save Method Preservation

Test each save method:
- `lora` - LoRA adapters only
- `merged_16bit` - Merged 16-bit model
- `merged_4bit` - Merged 4-bit model

Verify rerun uses same method.

### 19. Performance Test

**Steps**:
1. Rerun 10 jobs rapidly
2. Verify each gets unique ID and timestamp
3. Verify all enter queue properly
4. Verify no race conditions or data corruption

### 20. End-to-End Training Rerun

**Full integration test**:
1. Create and complete a small training job (e.g., 10 steps)
2. Verify job completes successfully
3. Rerun the job
4. Monitor new job through completion
5. Verify:
   - New model saved to different directory
   - Training metrics logged correctly
   - Carbon tracking works for rerun
   - WebSocket updates work
   - Job completes with same settings

## Manual Testing Checklist

- [ ] Rerun button visible on list page for finished jobs
- [ ] Rerun button visible on details page for finished jobs
- [ ] Rerun button NOT visible for running/queued jobs
- [ ] Confirmation dialogs work correctly
- [ ] Loading states display properly
- [ ] Redirects work after rerun
- [ ] Configuration display shows all parameters
- [ ] Badge colors are correct and readable
- [ ] Quality mode badge shows correct precision
- [ ] Early stopping badge shows correct patience
- [ ] Selective loss badge shows correct level
- [ ] Rerun From link works correctly
- [ ] New job name has timestamp suffix
- [ ] Output directory is different from original
- [ ] Queue integration works
- [ ] WebSocket updates work for rerun jobs
- [ ] Carbon tracking works for rerun jobs
- [ ] All hyperparameters preserved
- [ ] All LoRA parameters preserved
- [ ] Vision model reruns work
- [ ] Text model reruns work
- [ ] Failed job reruns work
- [ ] Cancelled job reruns work

## Automated Testing Script

```bash
#!/bin/bash

# Test script for training job rerun feature
API_BASE="http://localhost:8000/api/v1"

echo "Testing Training Job Rerun Feature"
echo "==================================="

# 1. Get list of training jobs
echo -e "\n1. Getting training jobs..."
JOBS=$(curl -s "$API_BASE/training/jobs")
echo "✓ Retrieved jobs"

# 2. Find a completed job
COMPLETED_JOB=$(echo "$JOBS" | jq -r '.items[] | select(.status == "completed") | .id' | head -n1)

if [ -z "$COMPLETED_JOB" ]; then
  echo "❌ No completed jobs found for testing"
  exit 1
fi

echo "✓ Found completed job: $COMPLETED_JOB"

# 3. Rerun the job
echo -e "\n2. Rerunning job..."
RERUN_RESPONSE=$(curl -s -X POST "$API_BASE/training/jobs/$COMPLETED_JOB/rerun")
NEW_JOB=$(echo "$RERUN_RESPONSE" | jq -r '.data.job_id')

if [ -z "$NEW_JOB" ] || [ "$NEW_JOB" = "null" ]; then
  echo "❌ Failed to create rerun job"
  echo "$RERUN_RESPONSE" | jq .
  exit 1
fi

echo "✓ Created rerun job: $NEW_JOB"

# 4. Verify new job exists
echo -e "\n3. Verifying new job..."
NEW_JOB_DATA=$(curl -s "$API_BASE/training/jobs/$NEW_JOB")
NEW_JOB_NAME=$(echo "$NEW_JOB_DATA" | jq -r '.name')

if [[ ! "$NEW_JOB_NAME" =~ _rerun_[0-9]{8}_[0-9]{6} ]]; then
  echo "❌ New job name doesn't have timestamp suffix: $NEW_JOB_NAME"
  exit 1
fi

echo "✓ New job has correct name format: $NEW_JOB_NAME"

# 5. Verify rerun metadata
RERUN_FROM=$(echo "$NEW_JOB_DATA" | jq -r '.rerun_from')
if [ "$RERUN_FROM" != "$COMPLETED_JOB" ]; then
  echo "❌ Rerun metadata incorrect: $RERUN_FROM != $COMPLETED_JOB"
  exit 1
fi

echo "✓ Rerun metadata correct"

# 6. Verify configuration preserved
ORIG_LR=$(curl -s "$API_BASE/training/jobs/$COMPLETED_JOB" | jq -r '.hyperparameters.learning_rate')
NEW_LR=$(echo "$NEW_JOB_DATA" | jq -r '.hyperparameters.learning_rate')

if [ "$ORIG_LR" != "$NEW_LR" ]; then
  echo "❌ Learning rate not preserved: $ORIG_LR != $NEW_LR"
  exit 1
fi

echo "✓ Hyperparameters preserved"

# 7. Test cannot rerun running job
echo -e "\n4. Testing error cases..."
if [ "$(echo "$NEW_JOB_DATA" | jq -r '.status')" = "queued" ] || [ "$(echo "$NEW_JOB_DATA" | jq -r '.status')" = "running" ]; then
  ERROR_RESPONSE=$(curl -s -X POST "$API_BASE/training/jobs/$NEW_JOB/rerun")
  if [ "$(echo "$ERROR_RESPONSE" | jq -r '.success')" = "true" ]; then
    echo "❌ Should not be able to rerun a running/queued job"
    exit 1
  fi
  echo "✓ Cannot rerun running/queued jobs"
fi

# 8. Test nonexistent job
ERROR_RESPONSE=$(curl -s -X POST "$API_BASE/training/jobs/fake-uuid-12345/rerun")
if [ "$(echo "$ERROR_RESPONSE" | jq -r '.success')" != "false" ]; then
  echo "❌ Should return error for nonexistent job"
  exit 1
fi

echo "✓ Error handling works correctly"

echo -e "\n==================================="
echo "✅ All tests passed!"
echo "New rerun job created: $NEW_JOB"
echo "Original job: $COMPLETED_JOB"
```

Save as `test_rerun_feature.sh` and run:
```bash
chmod +x test_rerun_feature.sh
./test_rerun_feature.sh
```

## Troubleshooting

### Issue: Rerun button not visible
**Check**: 
- Job status (should be completed/failed/cancelled)
- Frontend code loaded correctly
- Browser cache cleared

### Issue: Rerun fails with error
**Check**:
- API logs for detailed error
- Job exists in training_jobs.json
- Storage directory writable

### Issue: New job has same name as original
**Check**:
- Timestamp suffix generation in backend
- System clock is working correctly

### Issue: Configuration not preserved
**Check**:
- Original job has configuration in storage
- Deep copy is working (not just reference)
- All fields included in rerun endpoint code

### Issue: Queue not working
**Check**:
- Job queue service is running
- No errors in job queue logs
- Job status updating correctly

## Success Criteria

Feature is considered working correctly when:
1. ✅ All 20 test cases pass
2. ✅ Manual testing checklist completed
3. ✅ Automated test script passes
4. ✅ No errors in API logs
5. ✅ No errors in frontend console
6. ✅ UI is responsive and user-friendly
7. ✅ Performance is acceptable (<500ms for rerun creation)
8. ✅ Documentation is clear and complete
