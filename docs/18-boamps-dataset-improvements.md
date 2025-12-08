# BoAmps Dataset Section - Analysis & Improvement Recommendations

## Executive Summary

After reviewing the current BoAmps integration implementation against the official [BoAmps specification](https://github.com/Boavizta/BoAmps), significant improvements are needed in the **dataset section** to provide richer, more useful metadata for training reports. The current implementation is minimal and misses many optional fields that would greatly enhance report value for research and optimization.

## Current Implementation Issues

### 1. **Limited Dataset Metadata**

The current implementation (`model_garden/carbon/boamps.py`, lines 407-544) only captures basic dataset information:

```python
# Current minimal fields:
{
    "dataUsage": "input",           # Required - OK
    "dataType": "text",             # Required - OK  
    "source": "public",             # Optional - OK
    "sourceUri": "...",             # Optional - OK
    "dataFormat": "json",           # Optional - OK
    "dataSize": 0.5,                # Optional - OK but often missing
    "dataQuantity": 1000,           # Optional - OK but often missing
    "shape": "(224, 224)",          # Optional - rarely populated
    "owner": "..."                  # Optional - OK for HF datasets
}
```

### 2. **Missing Critical Dataset Fields**

According to the BoAmps `dataset_schema.json`, we're missing:

- **`items`** - Number of items in dataset (different from dataQuantity?)
- **`volume`** - Dataset size in bytes
- **`volumeUnit`** - Unit for volume (kilobyte, megabyte, gigabyte, etc.)
- **`inferenceProperties`** - Query/prompt metadata for inference datasets
- **`fileType`** - Specific file format (extensive enum of 100+ formats)

### 3. **Poor Dataset Size Tracking**

Currently, we only set `dataSize` (in GB) if `dataset_size` is in `job_config`, but:
- We don't reliably track actual dataset sizes
- We don't distinguish between `volume` (bytes) and `dataSize` (GB)
- No automatic dataset scanning/measurement during job execution

### 4. **Inadequate Vision Dataset Metadata**

For vision-language models, we need richer metadata:
- Image dimensions (width, height, channels)
- Number of images vs number of text samples
- Multimodal dataset structure (paired image-text)
- Vision-specific file types (jpg, png, webp, etc.)

### 5. **Missing Inference Dataset Details**

The BoAmps schema includes `inferenceProperties` with `Query` objects containing:
- `queryLength` - Token/character count of prompts
- `queryTokens` - Actual tokenized length
- `queryType` - Type of query (generation, classification, etc.)
- `queryText` - Sample query text

We don't capture any of this for inference jobs.

## BoAmps Dataset Schema Reference

From the BoAmps specification, the complete dataset schema is:

```json
{
  "dataUsage": "input|output",          // REQUIRED
  "dataType": "tabular|audio|boolean|image|video|object|text|token|word|other", // REQUIRED
  "fileType": "json|csv|parquet|jpg|png|...", // OPTIONAL - 100+ file formats
  "volume": 1048576,                     // OPTIONAL - size in bytes
  "volumeUnit": "megabyte|gigabyte|...", // OPTIONAL - if volume present
  "items": 10000,                        // OPTIONAL - number of items
  "dataQuantity": 10000,                 // OPTIONAL - alternative to items?
  "dataSize": 1.5,                       // OPTIONAL - size in GB
  "shape": "(224, 224, 3)",              // OPTIONAL - data dimensions
  "inferenceProperties": [               // OPTIONAL - for inference
    {
      "queryLength": 150,
      "queryTokens": 45,
      "queryType": "generation",
      "queryText": "Describe this image..."
    }
  ],
  "source": "public|private|other",      // OPTIONAL
  "sourceUri": "https://...",            // OPTIONAL
  "owner": "organization/user"           // OPTIONAL
}
```

## Recommended Improvements

### Priority 1: Core Dataset Metrics (HIGH IMPACT)

#### 1.1 Automatic Dataset Size Detection

**Implementation**: Add dataset analysis before training starts

```python
# In training.py / vision_training.py
def analyze_dataset(dataset_path: Path) -> dict:
    """Analyze dataset and return metadata."""
    stats = {
        "total_size_bytes": 0,
        "num_samples": 0,
        "file_type": None,
        "shape_info": {},
    }
    
    # For JSONL files
    if dataset_path.suffix == ".jsonl":
        stats["file_type"] = "json"
        stats["total_size_bytes"] = dataset_path.stat().st_size
        
        # Count lines for num_samples
        with open(dataset_path) as f:
            stats["num_samples"] = sum(1 for _ in f)
        
        # Analyze first sample for shape/structure
        with open(dataset_path) as f:
            first_sample = json.loads(f.readline())
            
            # For vision datasets
            if "image" in first_sample:
                # Load image to get dimensions
                img_path = Path(first_sample["image"])
                if img_path.exists():
                    from PIL import Image
                    img = Image.open(img_path)
                    stats["shape_info"] = {
                        "width": img.width,
                        "height": img.height,
                        "channels": len(img.getbands())
                    }
            
            # For text datasets
            if "text" in first_sample or "instruction" in first_sample:
                text = first_sample.get("text", first_sample.get("instruction", ""))
                stats["avg_text_length"] = len(text)
    
    # For HuggingFace datasets
    elif "/" in str(dataset_path):  # HF dataset identifier
        from datasets import load_dataset
        dataset = load_dataset(str(dataset_path), split="train")
        stats["num_samples"] = len(dataset)
        stats["file_type"] = "parquet"  # HF uses parquet internally
        
        # Estimate size from features
        # ... (analyze dataset.features)
    
    return stats

# Usage in training flow
dataset_stats = analyze_dataset(Path(dataset_path))
job_config["dataset_stats"] = dataset_stats
```

#### 1.2 Proper Volume/Size Distinction

The BoAmps schema has both `volume` (in bytes with `volumeUnit`) and `dataSize` (in GB). We should:

```python
def _build_datasets(self, ...):
    dataset_entry = {
        "dataUsage": "input",
        "dataType": primary_data_type,
    }
    
    # Add both volume and dataSize for clarity
    if "dataset_stats" in job_config:
        stats = job_config["dataset_stats"]
        
        # volume in bytes with volumeUnit
        size_bytes = stats.get("total_size_bytes", 0)
        if size_bytes > 0:
            # Convert to appropriate unit
            if size_bytes < 1024**2:  # < 1 MB
                dataset_entry["volume"] = size_bytes // 1024
                dataset_entry["volumeUnit"] = "kilobyte"
            elif size_bytes < 1024**3:  # < 1 GB
                dataset_entry["volume"] = size_bytes // (1024**2)
                dataset_entry["volumeUnit"] = "megabyte"
            else:
                dataset_entry["volume"] = size_bytes // (1024**3)
                dataset_entry["volumeUnit"] = "gigabyte"
            
            # Also add dataSize (in GB) for backwards compatibility
            dataset_entry["dataSize"] = round(size_bytes / (1024**3), 4)
        
        # items - actual number of samples
        if "num_samples" in stats:
            dataset_entry["items"] = stats["num_samples"]
```

#### 1.3 Rich File Type Information

Use the extensive BoAmps file type enum:

```python
def detect_file_type(dataset_path: str) -> str:
    """Detect detailed file type per BoAmps enum."""
    path = Path(dataset_path)
    
    # Extension-based detection
    ext_map = {
        ".jsonl": "json",
        ".json": "json",
        ".csv": "csv",
        ".tsv": "tsv",
        ".parquet": "parquet",
        ".arrow": "arrow",
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".webp": "webp",
        ".gif": "gif",
        ".mp4": "mp4",
        ".avi": "avi",
        ".wav": "wav",
        ".mp3": "mp3",
        ".flac": "flac",
        # ... add more from BoAmps enum
    }
    
    return ext_map.get(path.suffix.lower(), "other")
```

### Priority 2: Vision Dataset Enhancements (MEDIUM-HIGH IMPACT)

#### 2.1 Multimodal Dataset Structure

For vision-language training, we should track both modalities:

```python
def _build_datasets_vision(self, ...):
    """Build dataset entries for vision-language training."""
    datasets = []
    
    if "dataset_stats" in job_config:
        stats = job_config["dataset_stats"]
        
        # Image dataset entry
        image_entry = {
            "dataUsage": "input",
            "dataType": "image",
            "fileType": stats.get("image_format", "jpeg"),
            "items": stats.get("num_images", 0),
        }
        
        # Add image dimensions as shape
        if "shape_info" in stats:
            shape = stats["shape_info"]
            image_entry["shape"] = f"({shape['width']}, {shape['height']}, {shape['channels']})"
        
        datasets.append(image_entry)
        
        # Text dataset entry (paired with images)
        text_entry = {
            "dataUsage": "input",
            "dataType": "text",
            "fileType": "json",  # The JSONL containing text
            "items": stats.get("num_samples", 0),  # Same as images for paired data
        }
        
        # Add average text length info
        if "avg_text_length" in stats:
            text_entry["dataQuantity"] = stats["avg_text_length"]  # Avg chars/tokens
        
        datasets.append(text_entry)
    
    return datasets
```

#### 2.2 Image-Specific Metadata

Add vision-specific shape analysis:

```python
def analyze_vision_dataset(dataset_path: Path) -> dict:
    """Analyze vision dataset for detailed metadata."""
    from PIL import Image
    
    stats = {
        "num_images": 0,
        "image_formats": set(),
        "image_sizes": [],
        "total_image_bytes": 0,
    }
    
    # Read JSONL and analyze images
    with open(dataset_path) as f:
        for line in f:
            sample = json.loads(line)
            if "image" in sample:
                img_path = Path(sample["image"])
                if img_path.exists():
                    stats["num_images"] += 1
                    stats["total_image_bytes"] += img_path.stat().st_size
                    stats["image_formats"].add(img_path.suffix.lower())
                    
                    # Load to get dimensions
                    img = Image.open(img_path)
                    stats["image_sizes"].append((img.width, img.height))
    
    # Calculate averages
    if stats["image_sizes"]:
        avg_width = sum(w for w, h in stats["image_sizes"]) / len(stats["image_sizes"])
        avg_height = sum(h for w, h in stats["image_sizes"]) / len(stats["image_sizes"])
        stats["avg_image_shape"] = (int(avg_width), int(avg_height))
    
    # Determine most common format
    if stats["image_formats"]:
        stats["primary_format"] = max(stats["image_formats"], key=lambda x: x)
    
    return stats
```

### Priority 3: Inference Dataset Metadata (MEDIUM IMPACT)

#### 3.1 Implement inferenceProperties

For inference jobs, track query/prompt metadata:

```python
def _build_inference_properties(
    self,
    prompts: list[str],
    tokenizer=None
) -> list[dict]:
    """Build inferenceProperties for inference datasets."""
    properties = []
    
    for prompt in prompts[:10]:  # Sample first 10 prompts
        prop = {
            "queryLength": len(prompt),  # Character count
            "queryType": "generation",  # or "classification", "qa", etc.
        }
        
        # Add token count if tokenizer available
        if tokenizer:
            tokens = tokenizer.encode(prompt)
            prop["queryTokens"] = len(tokens)
        
        # Add sample text (truncated)
        if len(prompt) <= 200:
            prop["queryText"] = prompt
        else:
            prop["queryText"] = prompt[:200] + "..."
        
        properties.append(prop)
    
    return properties

# In _build_datasets for inference:
if task_stage == "inference" and "prompts" in job_config:
    output_entry["inferenceProperties"] = self._build_inference_properties(
        job_config["prompts"],
        job_config.get("tokenizer")
    )
```

### Priority 4: Dataset Validation & Quality (LOW-MEDIUM IMPACT)

#### 4.1 Dataset Quality Indicators

Add quality metrics to dataset metadata:

```python
def analyze_dataset_quality(dataset_path: Path) -> dict:
    """Analyze dataset quality and completeness."""
    quality = {
        "completeness": 0.0,  # % of samples with all fields
        "duplicates": 0,
        "avg_sample_quality": 0.0,
    }
    
    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f]
    
    # Check completeness
    required_fields = ["text", "response"]  # or ["image", "text", "response"]
    complete_samples = sum(
        1 for s in samples 
        if all(field in s and s[field] for field in required_fields)
    )
    quality["completeness"] = complete_samples / len(samples)
    
    # Check for duplicates (simple text-based)
    texts = [s.get("text", "") for s in samples]
    quality["duplicates"] = len(texts) - len(set(texts))
    
    return quality
```

## Implementation Plan

### Phase 1: Immediate Improvements (Week 1)
1. ✅ Add `analyze_dataset()` function to training modules
2. ✅ Implement proper `volume`/`volumeUnit` tracking
3. ✅ Add `items` field (actual sample count)
4. ✅ Improve `fileType` detection with BoAmps enum

### Phase 2: Vision Enhancements (Week 2)
1. ✅ Implement `analyze_vision_dataset()` 
2. ✅ Add multimodal dataset entries (separate image + text)
3. ✅ Enhance shape information with image dimensions
4. ✅ Track image format statistics

### Phase 3: Inference Metadata (Week 3)
1. ✅ Implement `inferenceProperties` for inference jobs
2. ✅ Track query lengths and token counts
3. ✅ Sample representative queries in reports
4. ✅ Add query type classification

### Phase 4: Validation & Quality (Week 4)
1. ✅ Add dataset quality metrics
2. ✅ Implement BoAmps schema validation
3. ✅ Add automated tests for all dataset scenarios
4. ✅ Documentation and examples

## Expected Impact

### Research Value
- **Dataset size studies**: Accurate size tracking enables research into optimal dataset sizes
- **Efficiency analysis**: Correlate dataset characteristics with energy consumption
- **Benchmark comparability**: Standardized metadata enables fair comparison across studies

### Operational Value
- **Resource planning**: Better estimates of training time/cost based on dataset size
- **Reproducibility**: Complete dataset metadata ensures experiments can be reproduced
- **Debugging**: Rich metadata helps diagnose training issues

### BoAmps Compliance
- **Schema validation**: Reports will pass BoAmps official validator
- **Open data contribution**: High-quality reports suitable for Boavizta open dataset
- **Community adoption**: Proper implementation encourages others to use the standard

## Code Changes Required

### Files to Modify

1. **`model_garden/carbon/boamps.py`** (Primary changes)
   - Lines 407-544: `_build_datasets()` - Expand dataset metadata
   - Add: `_build_inference_properties()` method
   - Add: `_analyze_dataset_shape()` method
   - Add: `_detect_file_type()` method

2. **`model_garden/training.py`** (Dataset analysis)
   - Add: `analyze_dataset()` function before training
   - Pass dataset stats to emissions tracker
   - Lines ~200-250: Add dataset metadata collection

3. **`model_garden/vision_training.py`** (Vision-specific)
   - Add: `analyze_vision_dataset()` function
   - Add: Image dimension analysis
   - Track multimodal dataset structure

4. **`model_garden/inference.py`** (Inference metadata)
   - Add: Query/prompt tracking
   - Add: Token count calculation
   - Pass inference properties to emissions tracker

5. **`tests/test_boamps.py`** (Test coverage)
   - Add: Dataset metadata tests
   - Add: Vision dataset tests
   - Add: Inference properties tests
   - Add: Schema validation tests

## Example: Enhanced Dataset Section

**Before (Current)**:
```json
{
  "dataset": [
    {
      "dataUsage": "input",
      "dataType": "text",
      "source": "public",
      "sourceUri": "https://huggingface.co/datasets/my-dataset"
    }
  ]
}
```

**After (Improved)**:
```json
{
  "dataset": [
    {
      "dataUsage": "input",
      "dataType": "image",
      "fileType": "jpeg",
      "volume": 2048,
      "volumeUnit": "megabyte",
      "dataSize": 2.0,
      "items": 10000,
      "shape": "(224, 224, 3)",
      "source": "public",
      "sourceUri": "https://huggingface.co/datasets/my-dataset",
      "owner": "organization"
    },
    {
      "dataUsage": "input",
      "dataType": "text",
      "fileType": "json",
      "volume": 128,
      "volumeUnit": "megabyte",
      "items": 10000,
      "dataQuantity": 150,
      "source": "public",
      "sourceUri": "https://huggingface.co/datasets/my-dataset"
    }
  ]
}
```

## References

- **BoAmps GitHub**: https://github.com/Boavizta/BoAmps
- **BoAmps Schema**: https://github.com/Boavizta/BoAmps/blob/main/model/dataset_schema.json
- **BoAmps Examples**: https://github.com/Boavizta/BoAmps/tree/main/examples
- **BoAmps Validator**: https://github.com/Boavizta/BoAmps/tree/main/tools/schema_validator
- **Open Dataset**: https://huggingface.co/datasets/boavizta/open_data_boamps

## Conclusion

The dataset section is critical for making BoAmps reports valuable for research and operational use. By implementing these improvements, Model Garden will:

1. **Generate higher-quality reports** suitable for the Boavizta open dataset
2. **Enable better research** into AI efficiency and optimization
3. **Improve reproducibility** of training experiments
4. **Enhance operational insights** for resource planning

The improvements are backward-compatible and can be implemented incrementally over 4 weeks with minimal disruption to existing functionality.
