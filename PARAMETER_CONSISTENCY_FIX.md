# Parameter Consistency Fix

## Summary
Fixed critical parameter consistency issues where training job creation was missing early stopping parameters and the API response model was stripping out all new parameters.

## Issues Found

### 1. Missing Early Stopping Parameters in Job Creation
**Location:** `model_garden/api.py` - `create_training_job()` function (line ~1790-1803)

**Problem:** When creating a new training job, the early stopping parameters from `TrainingJobRequest` were not being stored in the job record. This meant:
- Early stopping settings were ignored when creating jobs
- Jobs couldn't be rerun with original early stopping configuration
- The WebUI couldn't display early stopping settings

**Parameters Missing:**
- `early_stopping_enabled`
- `early_stopping_patience`
- `early_stopping_threshold`

### 2. Incomplete TrainingJobInfo Pydantic Model
**Location:** `model_garden/api.py` - `TrainingJobInfo` class (line 93-117)

**Problem:** The Pydantic response model was missing ALL new parameter fields, causing them to be stripped from API responses even when stored in the job record. This broke:
- Job details page display
- Rerun functionality (couldn't see what to clone)
- Any API consumers expecting these fields

**Fields Missing:**
- Selective loss: `selective_loss`, `selective_loss_level`, `selective_loss_schema_keys`, `selective_loss_masking_start_epoch`, `selective_loss_verbose`
- Quality settings: `quality_mode`, `load_in_16bit`, `load_in_8bit`
- Early stopping: `early_stopping_enabled`, `early_stopping_patience`, `early_stopping_threshold`
- Rerun metadata: `rerun_from`, `rerun_from_name`
- Queue position: `queue_position`

## Verification Process

### Step 1: Frontend Form Analysis ✅
**File:** `frontend/src/routes/training/new/+page.svelte`

**Findings:**
- Lines 1650-1750: Complete early stopping UI with enabled checkbox, patience, and threshold inputs
- Lines 2020-2200: Complete selective loss UI for vision models (level selector, schema keys, masking start epoch slider, verbose checkbox)
- Lines 280-320: `handleSubmit()` correctly sends all parameters to API client
- All parameters properly initialized in form state

**Conclusion:** Frontend captures and sends all parameters correctly ✅

### Step 2: API Client Analysis ✅
**File:** `frontend/src/lib/api/client.ts`

**Findings:**
- `TrainingJob` interface includes all new fields
- `createTrainingJob()` sends complete request payload
- `rerunTrainingJob()` properly clones jobs

**Conclusion:** TypeScript client handles all parameters correctly ✅

### Step 3: Backend Request Model Analysis ✅
**File:** `model_garden/api.py` - `TrainingJobRequest` (lines 64-90)

**Findings:**
- All parameters present with correct types and defaults
- Selective loss: 5 fields (selective_loss, level, schema_keys, masking_start_epoch, verbose)
- Quality settings: 3 fields (quality_mode, load_in_16bit, load_in_8bit)
- Early stopping: 3 fields (enabled, patience, threshold)

**Conclusion:** Request validation accepts all parameters ✅

### Step 4: Job Creation Implementation Analysis ❌ → ✅
**File:** `model_garden/api.py` - `create_training_job()` (lines 1750-1840)

**Initial Findings:**
- Selective loss parameters: ✅ Stored correctly (lines 1793-1797)
- Quality settings: ✅ Stored correctly (lines 1799-1801)
- Early stopping: ❌ **MISSING** - not stored in job_info

**Impact:**
- Jobs created without early stopping settings even when provided
- Rerun would fail to clone early stopping configuration
- Training would ignore early stopping parameters

**Fix Applied:**
Added early stopping parameters to job_info dictionary:
```python
# Early stopping settings
"early_stopping_enabled": job_request.early_stopping_enabled,
"early_stopping_patience": job_request.early_stopping_patience,
"early_stopping_threshold": job_request.early_stopping_threshold,
```

### Step 5: Response Model Analysis ❌ → ✅
**File:** `model_garden/api.py` - `TrainingJobInfo` (lines 93-117)

**Initial Findings:**
- Response model only had legacy fields
- All new parameters missing from Pydantic model
- Pydantic would strip these fields from responses

**Impact:**
- API responses missing all new parameters
- Frontend couldn't display selective loss, early stopping, or quality settings
- Rerun feature couldn't see configuration to clone
- Job details page showed incomplete information

**Fix Applied:**
Added all missing fields to `TrainingJobInfo`:
```python
# Selective loss settings
selective_loss: Optional[bool] = False
selective_loss_level: Optional[str] = "conservative"
selective_loss_schema_keys: Optional[List[str]] = None
selective_loss_masking_start_epoch: Optional[float] = 0.0
selective_loss_verbose: Optional[bool] = False

# Quality settings
quality_mode: Optional[bool] = False
load_in_16bit: Optional[bool] = False
load_in_8bit: Optional[bool] = False

# Early stopping settings
early_stopping_enabled: Optional[bool] = False
early_stopping_patience: Optional[int] = 3
early_stopping_threshold: Optional[float] = 0.0

# Rerun metadata
rerun_from: Optional[str] = None
rerun_from_name: Optional[str] = None
queue_position: Optional[int] = None
```

### Step 6: Rerun Endpoint Analysis ✅
**File:** `model_garden/api.py` - `rerun_training_job()` (lines 1960-2070)

**Findings:**
- Already correctly clones all parameters including early stopping (lines 2036-2038)
- Properly handles selective loss, quality settings, and metadata
- Creates new job with timestamp suffix

**Conclusion:** Rerun implementation was already correct ✅

## Parameter Flow Verification

### Complete Flow
1. **Frontend Form** → User inputs all parameters ✅
2. **API Client** → Sends parameters in request ✅
3. **Backend Request Model** → Validates incoming parameters ✅
4. **Job Creation** → Stores parameters in job record ✅ (fixed)
5. **Job Storage** → Persists to `training_jobs.json` ✅
6. **Job Retrieval** → Returns job via API ✅ (fixed)
7. **Response Model** → Includes all fields ✅ (fixed)
8. **Frontend Display** → Shows parameters in UI ✅

### Parameter Coverage Matrix

| Parameter | Form | API Client | Request Model | Job Creation | Response Model | Status |
|-----------|------|------------|---------------|--------------|----------------|--------|
| **Selective Loss** |
| selective_loss | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| selective_loss_level | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| selective_loss_schema_keys | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| selective_loss_masking_start_epoch | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| selective_loss_verbose | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| **Quality Settings** |
| quality_mode | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| load_in_16bit | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| load_in_8bit | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| **Early Stopping** |
| early_stopping_enabled | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| early_stopping_patience | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| early_stopping_threshold | ✅ | ✅ | ✅ | ✅ | ✅ | Fixed |
| **Rerun Metadata** |
| rerun_from | N/A | ✅ | N/A | N/A | ✅ | Fixed |
| rerun_from_name | N/A | ✅ | N/A | N/A | ✅ | Fixed |

## Files Modified

### 1. model_garden/api.py
**Changes:**
1. Updated `TrainingJobInfo` class to include all missing fields (lines 93-134)
2. Updated `create_training_job()` to store early stopping parameters (lines 1803-1806)

**Impact:**
- All parameters now properly stored when creating jobs
- All parameters returned in API responses
- Job details page can display all settings
- Rerun feature has complete configuration to clone

## Testing Recommendations

### 1. Job Creation Test
```bash
# Create a job with all parameters
curl -X POST http://localhost:8000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-all-params",
    "base_model": "unsloth/tinyllama-bnb-4bit",
    "dataset_path": "data/sample.jsonl",
    "validation_dataset_path": "data/val.jsonl",
    "output_dir": "models/test",
    "early_stopping_enabled": true,
    "early_stopping_patience": 5,
    "early_stopping_threshold": 0.01,
    "selective_loss": true,
    "selective_loss_level": "aggressive",
    "quality_mode": true,
    "load_in_16bit": true
  }'

# Verify all parameters are stored
curl http://localhost:8000/api/v1/training/jobs/{job_id}
```

### 2. Rerun Test
```bash
# Rerun a job with all parameters
curl -X POST http://localhost:8000/api/v1/training/jobs/{job_id}/rerun

# Verify new job has all cloned parameters
curl http://localhost:8000/api/v1/training/jobs/{new_job_id}
```

### 3. WebUI Test
1. Create a training job via UI with all parameters enabled
2. Check job details page displays all sections:
   - Hyperparameters
   - LoRA Configuration
   - Quality Settings
   - Early Stopping (when validation dataset provided)
   - Selective Loss (for vision models)
3. Rerun the job and verify cloned settings

## Root Cause Analysis

### Why This Happened
1. **Incremental Feature Development**: Parameters were added to request model but not consistently propagated to job storage and response model
2. **Pydantic Strictness**: Pydantic models strip unknown fields by default, silently hiding the problem
3. **Missing Validation**: No end-to-end tests verifying parameter flow from form to storage to display

### Prevention Measures
1. **Checklist for New Parameters**: When adding parameters, update:
   - [ ] Request model (`TrainingJobRequest`)
   - [ ] Response model (`TrainingJobInfo`)
   - [ ] Job creation storage (`create_training_job`)
   - [ ] Rerun cloning (`rerun_training_job`)
   - [ ] Frontend form
   - [ ] Job details display
   - [ ] API client interface

2. **Testing Strategy**: Add integration tests that verify:
   - Parameters survive round-trip (create → retrieve)
   - Rerun correctly clones all parameters
   - API responses include all expected fields

3. **Code Review**: Check for Pydantic model completeness when adding fields to job records

## Status
✅ **FIXED** - All parameters now consistently stored and returned throughout the stack.

## Related Issues
- Selective loss parameters not visible in WebUI (resolved)
- Early stopping settings not preserved (resolved)
- Parameter inconsistency between frontend and backend (resolved)
