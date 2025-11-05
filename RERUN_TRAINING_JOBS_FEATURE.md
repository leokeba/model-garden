# Training Job Rerun Feature - Implementation Summary

## Overview
Implemented a comprehensive feature to rerun past training jobs with all their original configurations. This allows users to retry failed jobs, rerun successful experiments, or create variations of previous training runs.

## Implementation Details

### 1. Backend API Endpoint
**File**: `model_garden/api.py`

Created a new POST endpoint: `/api/v1/training/jobs/{job_id}/rerun`

**Features**:
- Clones all configuration from the original job
- Creates a new job with a timestamp suffix (e.g., `model_name_rerun_20251022_143025`)
- Generates a unique output directory to avoid conflicts
- Prevents rerunning jobs that are currently running or queued
- Preserves all training parameters including:
  - Base model and datasets (training + validation)
  - Hyperparameters (learning rate, epochs, batch size, etc.)
  - LoRA configuration (rank, alpha, dropout, target modules, etc.)
  - Quality settings (quality_mode, load_in_16bit, load_in_8bit)
  - Early stopping settings (enabled, patience, threshold)
  - Selective loss settings (enabled, level, schema keys, masking start epoch, verbose)
  - Model type (text/vision) and save method
  - HuggingFace Hub flags (from_hub, validation_from_hub)
- Adds metadata to track the rerun relationship:
  - `rerun_from`: Original job ID
  - `rerun_from_name`: Original job name

**Response**:
```json
{
  "success": true,
  "data": {
    "job_id": "new-uuid",
    "original_job_id": "original-uuid",
    "queue_position": 1,
    "name": "model_name_rerun_20251022_143025"
  },
  "message": "Training job rerun created and queued for execution"
}
```

### 2. Frontend API Client
**File**: `frontend/src/lib/api/client.ts`

**Changes**:
1. Added `rerunTrainingJob()` method to APIClient class
2. Extended `TrainingJob` TypeScript interface to include:
   - Quality settings: `quality_mode`, `load_in_16bit`, `load_in_8bit`
   - Early stopping: `early_stopping_enabled`, `early_stopping_patience`, `early_stopping_threshold`
   - Selective loss: `selective_loss`, `selective_loss_level`, `selective_loss_schema_keys`, `selective_loss_masking_start_epoch`, `selective_loss_verbose`
   - Rerun metadata: `rerun_from`, `rerun_from_name`

### 3. Job Details Page UI
**File**: `frontend/src/routes/training/[id]/+page.svelte`

**Changes**:
1. Added `rerunning` state variable to track rerun operation
2. Added `rerunJob()` async function with confirmation dialog
3. Added "🔄 Rerun Training" button in the Actions card (visible for completed/failed/cancelled jobs)
4. Enhanced Configuration display with additional sections:
   - **Quality Mode**: Shows precision level (16-bit, 8-bit, or 4-bit default) with color-coded badges
   - **Early Stopping**: Displays if enabled with patience value
   - **Selective Loss**: Shows level (conservative/moderate/aggressive) if enabled
   - **Rerun From**: Clickable link to the original job if this is a rerun
5. All sections use consistent badge styling for visual clarity

### 4. Training List Page UI
**File**: `frontend/src/routes/training/+page.svelte`

**Changes**:
1. Added `rerunningJobId` state to track which job is being rerun
2. Added `handleRerun()` async function with confirmation dialog
3. Replaced simple action buttons with a more comprehensive layout:
   - Running/Queued jobs: "Cancel" button (red/danger)
   - Completed/Failed/Cancelled jobs: "🔄 Rerun" button (blue/primary) + "Delete" button (red/danger)
4. Rerun button shows loading state while creating the new job
5. Automatically navigates to the new job page after successful rerun

## User Experience Flow

### From Job Details Page:
1. User views a completed/failed/cancelled training job
2. User sees all configuration parameters clearly displayed
3. User clicks "🔄 Rerun Training" button in Actions section
4. Confirmation dialog shows: "Rerun training job 'NAME'? This will create a new training job with the same configuration. The original job will remain unchanged."
5. Upon confirmation, a new job is created and user is redirected to the new job's page
6. User can monitor the new training run with real-time updates

### From Training List Page:
1. User browses list of training jobs
2. For completed/failed/cancelled jobs, both "🔄 Rerun" and "Delete" buttons are visible
3. User clicks "🔄 Rerun" button
4. Confirmation dialog appears
5. Upon confirmation, user is redirected to the new job's details page
6. Original job remains in the list and is linked from the new job via "Rerun From" field

## Configuration Visibility

All relevant training parameters are now clearly visible in the job details UI:

### Basic Settings
- Model Type (text/vision)
- Validation Dataset (enabled/not provided)
- Save Method (lora/merged_16bit/merged_4bit)

### Quality Settings
- Quality Mode badge (if enabled)
- Precision level (16-bit/8-bit/4-bit) with color coding:
  - 16-bit: Blue badge
  - 8-bit: Yellow badge
  - 4-bit: Gray text (default)

### Early Stopping
- Enabled status with patience value (if enabled)
- Purple badge for visual distinction

### Selective Loss
- Enabled status with level indicator
- Indigo badge for visual distinction

### Hyperparameters
- Learning Rate, Epochs, Batch Size
- Max Steps, Gradient Accumulation, Warmup Steps
- Optimizer, LR Scheduler, Weight Decay
- Max Grad Norm, Eval Strategy, Eval Steps
- And many more in collapsible "Advanced Settings"

### LoRA Configuration
- Rank (r), Alpha, Dropout, Bias
- RSLoRA status, Task Type, Target Modules
- Gradient Checkpointing, Random Seed

## Technical Implementation Notes

### Backend Considerations
1. **Path Resolution**: Uses the existing `resolve_model_path()` to handle model paths correctly
2. **UUID Generation**: Each rerun gets a unique UUID to prevent conflicts
3. **Timestamp Suffix**: Adds readable timestamp to job name for easy identification
4. **Deep Copy**: Uses `.copy()` on dictionaries to prevent reference issues
5. **Queue Integration**: Properly integrates with the job queue system
6. **Validation**: Checks job status before allowing rerun
7. **Persistence**: Saves to storage immediately to survive server restarts

### Frontend Considerations
1. **State Management**: Uses Svelte 5's `$state` runes for reactive updates
2. **Loading States**: Shows loading indicators during async operations
3. **Error Handling**: Catches and displays errors gracefully
4. **Navigation**: Automatic redirect to new job page after successful rerun
5. **Visual Feedback**: Uses badges and color coding for quick parameter scanning
6. **Confirmation Dialogs**: Prevents accidental reruns with clear messaging

## Testing Recommendations

To verify the feature works correctly, test:

1. **Text Model Rerun**: Rerun a completed text-only training job
2. **Vision Model Rerun**: Rerun a completed vision-language model training job
3. **Failed Job Rerun**: Rerun a failed job to retry with same settings
4. **Quality Mode**: Verify quality_mode, load_in_16bit, load_in_8bit are preserved
5. **Early Stopping**: Verify early stopping settings are preserved
6. **Selective Loss**: Verify selective loss settings are preserved
7. **Validation Dataset**: Verify jobs with validation datasets preserve both datasets
8. **HuggingFace Hub**: Verify from_hub and validation_from_hub flags are preserved
9. **Queue Position**: Verify new job is properly queued and shows position
10. **Output Directory**: Verify new job creates a separate output directory
11. **Metadata**: Verify "Rerun From" link works and points to original job
12. **UI Display**: Verify all parameters are visible in the Configuration section

## Benefits

1. **Experiment Reproducibility**: Easily rerun successful experiments
2. **Failure Recovery**: Quickly retry failed jobs without manual reconfiguration
3. **Parameter Visibility**: All settings are clearly displayed for transparency
4. **Time Saving**: No need to manually enter all parameters again
5. **Audit Trail**: Clear linkage between original and rerun jobs
6. **User-Friendly**: Simple one-click operation with clear confirmations
7. **Flexible**: Works with all training configurations (text, vision, quality modes, etc.)

## Future Enhancements

Potential improvements for future iterations:
1. Allow editing parameters before rerunning (e.g., change learning rate)
2. Bulk rerun multiple jobs
3. Create rerun templates/presets
4. Show rerun history/chain visualization
5. Compare metrics between original and rerun jobs
6. Export/import job configurations
7. Schedule reruns for later execution
8. Add tags/labels to organize related reruns

## Files Modified

1. `model_garden/api.py` - Added rerun endpoint
2. `frontend/src/lib/api/client.ts` - Added rerun method and extended interface
3. `frontend/src/routes/training/[id]/+page.svelte` - Added rerun UI to details page
4. `frontend/src/routes/training/+page.svelte` - Added rerun UI to list page

## Deployment Notes

- No database migration required (uses existing JSON storage)
- No environment variable changes needed
- Frontend requires rebuild: `npm run build` in frontend directory
- Backend will pick up changes on restart (systemd service restart if deployed)
- Feature is backward compatible with existing jobs
