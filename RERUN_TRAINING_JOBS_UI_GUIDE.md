# Training Job Rerun Feature - UI Guide

## UI Components Overview

### 1. Training List Page (`/training`)

#### Before (Completed/Failed Jobs)
```
┌─────────────────────────────────────────────────────────────────┐
│ Job Name: my-fine-tuned-model                                    │
│ Status: [Completed]                                              │
│                                                                   │
│ Base Model: unsloth/llama-3-8b-bnb-4bit                         │
│ Dataset: ./data/train.jsonl                                      │
│ Created: 10/22/2025 2:30 PM                                     │
│                                                                   │
│                                   [Details] [Delete]             │
└─────────────────────────────────────────────────────────────────┘
```

#### After (Completed/Failed Jobs)
```
┌─────────────────────────────────────────────────────────────────┐
│ Job Name: my-fine-tuned-model                                    │
│ Status: [Completed]                                              │
│                                                                   │
│ Base Model: unsloth/llama-3-8b-bnb-4bit                         │
│ Dataset: ./data/train.jsonl                                      │
│ Created: 10/22/2025 2:30 PM                                     │
│                                                                   │
│                        [Details] [🔄 Rerun] [Delete]            │
└─────────────────────────────────────────────────────────────────┘
```

**Changes**:
- Added "🔄 Rerun" button between Details and Delete
- Button is primary style (blue) to encourage use
- Shows "Starting..." when clicked
- Only visible for completed/failed/cancelled jobs

#### Running/Queued Jobs (Unchanged)
```
┌─────────────────────────────────────────────────────────────────┐
│ Job Name: training-in-progress                                   │
│ Status: [Running]                                                │
│ Progress: ████████░░░░░░░░░░░░░░  350/1000 steps                │
│                                                                   │
│                                     [Details] [Cancel]           │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Job Details Page (`/training/{id}`)

#### Configuration Section - Basic Info
```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Model Type:              text                                    │
│ Validation Dataset:      [✓ Enabled]                            │
│ Save Method:             merged_16bit                            │
│ Quality Mode:            [✓ Enabled (16-bit)]                   │
│ Early Stopping:          [✓ Enabled (patience: 3)]              │
│ Selective Loss:          [✓ conservative]                       │
│ Rerun From:              original-job-name (clickable)          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**Badge Color Coding**:
- **16-bit Quality**: Blue badge (`bg-blue-100 text-blue-800`)
- **8-bit Quality**: Yellow badge (`bg-yellow-100 text-yellow-800`)
- **Validation Enabled**: Green badge (`bg-green-100 text-green-800`)
- **Early Stopping**: Purple badge (`bg-purple-100 text-purple-800`)
- **Selective Loss**: Indigo badge (`bg-indigo-100 text-indigo-800`)

#### Actions Section
```
┌─────────────────────────────────────────────────────────────────┐
│ Actions                                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ [           Refresh Status            ]  (primary)               │
│                                                                   │
│ [        🔄 Rerun Training           ]  (primary, new!)          │
│                                                                   │
│ [           View Model               ]  (secondary)              │
│                                                                   │
│ [        Start New Job               ]  (secondary)              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**For Running Jobs**:
```
┌─────────────────────────────────────────────────────────────────┐
│ Actions                                                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ [           Refresh Status            ]  (primary)               │
│                                                                   │
│ [         ⏸️ Stop Early              ]  (warning/yellow)         │
│                                                                   │
│ [        Cancel Training             ]  (danger/red)             │
│                                                                   │
│ [        Start New Job               ]  (secondary)              │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Confirmation Dialogs

#### Rerun from List Page
```
┌─────────────────────────────────────────────────┐
│  Rerun training job "my-fine-tuned-model"?      │
│                                                  │
│  This will create a new training job with       │
│  the same configuration.                        │
│                                                  │
│              [Cancel]    [OK]                   │
└─────────────────────────────────────────────────┘
```

#### Rerun from Details Page
```
┌─────────────────────────────────────────────────┐
│  Rerun training job "my-fine-tuned-model"?      │
│                                                  │
│  This will create a new training job with       │
│  the same configuration. The original job       │
│  will remain unchanged.                         │
│                                                  │
│              [Cancel]    [OK]                   │
└─────────────────────────────────────────────────┘
```

### 4. Parameter Visibility Examples

#### Vision Model with Quality Mode
```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Model Type:              vision                                  │
│ Validation Dataset:      [✓ Enabled]                            │
│ Save Method:             merged_16bit                            │
│ Quality Mode:            [✓ Enabled (16-bit)]                   │
│                                                                   │
│ ─────────────────────────────────────────────────────────────── │
│                                                                   │
│ Training Hyperparameters                                         │
│                                                                   │
│ Learning Rate:    2e-5          Epochs:           3              │
│ Batch Size:       1             Max Steps:        Auto           │
│ Gradient Accum:   8             Warmup Steps:     10             │
│ Optimizer:        adamw_torch   LR Scheduler:     cosine         │
│ Weight Decay:     0.01          Max Grad Norm:    1.0            │
│ Eval Strategy:    steps         Eval Steps:       50             │
│                                                                   │
│ ─────────────────────────────────────────────────────────────── │
│                                                                   │
│ LoRA Configuration                                               │
│                                                                   │
│ Rank (r):         32            Alpha:            32             │
│ Dropout:          0.0           Bias:             none           │
│ RSLoRA:           Enabled       Task Type:        CAUSAL_LM      │
│ Target Modules:   q_proj, k_proj, v_proj, o_proj, ...          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

#### Text Model with Early Stopping and Selective Loss
```
┌─────────────────────────────────────────────────────────────────┐
│ Configuration                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Model Type:              text                                    │
│ Validation Dataset:      [✓ Enabled]                            │
│ Save Method:             lora                                    │
│ Quality Mode:            8-bit Precision                         │
│ Early Stopping:          [✓ Enabled (patience: 5)]              │
│ Selective Loss:          [✓ moderate]                           │
│                                                                   │
│ ─────────────────────────────────────────────────────────────── │
│                                                                   │
│ Training Hyperparameters                                         │
│                                                                   │
│ Learning Rate:    2e-4          Epochs:           5              │
│ Batch Size:       2             Max Steps:        -1             │
│ Gradient Accum:   4             Warmup Steps:     50             │
│ Optimizer:        adamw_8bit    LR Scheduler:     linear         │
│ Weight Decay:     0.01          Max Grad Norm:    1.0            │
│ Eval Strategy:    steps         Eval Steps:       100            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5. Rerun Metadata Display

When viewing a rerun job:
```
┌─────────────────────────────────────────────────────────────────┐
│ Job Information                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Job ID:          abc-123-def-456                                 │
│ Status:          [Running]                                       │
│ Model Name:      my-model_rerun_20251022_143025                 │
│ Base Model:      unsloth/llama-3-8b-bnb-4bit                    │
│ Dataset:         ./data/train.jsonl                              │
│ Output Dir:      ./models/my-model_rerun_20251022_143025        │
│ Created:         10/22/2025 2:30:25 PM                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ Configuration                                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│ Model Type:              text                                    │
│ Validation Dataset:      [✓ Enabled]                            │
│ Save Method:             merged_16bit                            │
│ Quality Mode:            [✓ Enabled (16-bit)]                   │
│ Rerun From:              my-model (clickable link)              │
│                          ↑ Links back to original job            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## User Flow Diagrams

### Rerun from List
```
Training List Page
       │
       ├─ User sees completed job with [🔄 Rerun] button
       │
       ├─ User clicks [🔄 Rerun]
       │
       ├─ Confirmation dialog appears
       │
       ├─ User clicks [OK]
       │
       ├─ Button shows "Starting..."
       │
       ├─ API creates new job with timestamp suffix
       │
       ├─ User is redirected to new job details page
       │
       └─ New job starts training with same config
```

### Rerun from Details
```
Job Details Page (completed job)
       │
       ├─ User reviews all configuration parameters
       │
       ├─ User sees [🔄 Rerun Training] in Actions
       │
       ├─ User clicks [🔄 Rerun Training]
       │
       ├─ Confirmation dialog with detailed info
       │
       ├─ User confirms
       │
       ├─ New job is created
       │
       ├─ User is redirected to new job page
       │
       └─ New job shows "Rerun From: original-job-name"
```

## Responsive Design Notes

- On mobile, buttons stack vertically
- On desktop, buttons are inline with appropriate spacing
- Badge text truncates gracefully on small screens
- Configuration grid adapts to 1 column on mobile, 2 columns on desktop
- Action buttons remain full-width on mobile for easier tapping

## Accessibility

- All buttons have clear labels
- Color is not the only indicator (text + icons used)
- Loading states announced to screen readers
- Confirmation dialogs are keyboard accessible
- Links have proper ARIA labels
- Badge text is readable at all sizes

## Visual Hierarchy

1. **Primary Actions** (Blue): Rerun Training, Refresh Status
2. **Warning Actions** (Yellow): Stop Early
3. **Danger Actions** (Red): Cancel, Delete
4. **Secondary Actions** (Gray): View Model, Start New Job

Badge hierarchy:
1. **Status badges**: Largest, most prominent
2. **Feature badges**: Medium, grouped by category
3. **Info badges**: Small, supplementary information
