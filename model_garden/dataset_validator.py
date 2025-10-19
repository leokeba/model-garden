"""Dataset validation and analysis utilities."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter


@dataclass
class DatasetStats:
    """Dataset statistics."""
    total_rows: int
    format: str
    fields: List[str]
    field_types: Dict[str, str]
    missing_fields: Dict[str, int]
    sample_rows: List[Dict[str, Any]]
    file_size_bytes: int
    validation_errors: List[str]
    warnings: List[str]
    
    # Text-specific stats
    avg_input_length: Optional[float] = None
    avg_output_length: Optional[float] = None
    total_tokens_estimate: Optional[int] = None
    
    # Vision-specific stats
    has_images: bool = False
    image_count: int = 0
    image_paths: Optional[List[str]] = None
    statistics: Optional[Dict[str, Any]] = None


class DatasetValidator:
    """Validate and analyze datasets for training."""
    
    # Standard schemas for different dataset types
    TEXT_SCHEMA = {
        "required_fields": ["instruction", "output"],
        "optional_fields": ["input", "context"],
        "field_types": {
            "instruction": str,
            "input": str,
            "output": str,
            "context": str
        }
    }
    
    VISION_SCHEMA = {
        "required_fields": ["text", "image", "response"],
        "optional_fields": ["metadata"],
        "field_types": {
            "text": str,
            "image": str,
            "response": str,
            "metadata": dict
        }
    }
    
    ALPACA_SCHEMA = {
        "required_fields": ["instruction", "output"],
        "optional_fields": ["input"],
        "field_types": {
            "instruction": str,
            "input": str,
            "output": str
        }
    }
    
    @staticmethod
    def detect_format(file_path: Path) -> str:
        """Detect dataset format from file extension."""
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            return "jsonl"
        elif suffix == ".json":
            return "json"
        elif suffix == ".csv":
            return "csv"
        else:
            return "unknown"
    
    @staticmethod
    def detect_schema_type(sample_data: List[Dict[str, Any]]) -> str:
        """Detect which schema the dataset follows."""
        if not sample_data:
            return "unknown"
        
        first_row = sample_data[0]
        fields = set(first_row.keys())
        
        # Check for vision schema
        if "image" in fields and "text" in fields and "response" in fields:
            return "vision"
        
        # Check for Alpaca schema
        if "instruction" in fields and "output" in fields:
            return "alpaca"
        
        # Check for text schema
        if "instruction" in fields or ("input" in fields and "output" in fields):
            return "text"
        
        return "custom"
    
    @staticmethod
    def load_dataset(file_path: Path, max_rows: int = 1000) -> Tuple[List[Dict[str, Any]], str]:
        """
        Load dataset from file.
        
        Args:
            file_path: Path to dataset file
            max_rows: Maximum rows to load (for preview/validation)
            
        Returns:
            Tuple of (data rows, format)
        """
        format_type = DatasetValidator.detect_format(file_path)
        data = []
        
        try:
            if format_type == "jsonl":
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= max_rows:
                            break
                        if line.strip():
                            data.append(json.loads(line))
            
            elif format_type == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if isinstance(json_data, list):
                        data = json_data[:max_rows]
                    else:
                        raise ValueError("JSON file must contain an array of objects")
            
            elif format_type == "csv":
                with open(file_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for i, row in enumerate(reader):
                        if i >= max_rows:
                            break
                        data.append(row)
            
            else:
                raise ValueError(f"Unsupported format: {format_type}")
        
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {str(e)}")
        
        return data, format_type
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (1 token ≈ 4 characters)."""
        if not text:
            return 0
        return len(text) // 4
    
    @staticmethod
    def validate_dataset(
        file_path: Path,
        schema_type: Optional[str] = None,
        max_sample: int = 100
    ) -> DatasetStats:
        """
        Validate and analyze a dataset.
        
        Args:
            file_path: Path to dataset file
            schema_type: Expected schema type (auto-detect if None)
            max_sample: Number of rows to sample for analysis
            
        Returns:
            DatasetStats with validation results
        """
        errors = []
        warnings = []
        
        # Check file exists
        if not file_path.exists():
            errors.append(f"File not found: {file_path}")
            return DatasetStats(
                total_rows=0,
                format="unknown",
                fields=[],
                field_types={},
                missing_fields={},
                sample_rows=[],
                file_size_bytes=0,
                validation_errors=errors,
                warnings=warnings
            )
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Load dataset
        try:
            data, format_type = DatasetValidator.load_dataset(file_path, max_rows=max_sample)
        except Exception as e:
            errors.append(str(e))
            return DatasetStats(
                total_rows=0,
                format="unknown",
                fields=[],
                field_types={},
                missing_fields={},
                sample_rows=[],
                file_size_bytes=file_size,
                validation_errors=errors,
                warnings=warnings
            )
        
        if not data:
            errors.append("Dataset is empty")
            return DatasetStats(
                total_rows=0,
                format=format_type,
                fields=[],
                field_types={},
                missing_fields={},
                sample_rows=[],
                file_size_bytes=file_size,
                validation_errors=errors,
                warnings=warnings
            )
        
        # Detect schema if not provided
        if not schema_type:
            schema_type = DatasetValidator.detect_schema_type(data)
        
        # Get schema
        schema = {
            "vision": DatasetValidator.VISION_SCHEMA,
            "alpaca": DatasetValidator.ALPACA_SCHEMA,
            "text": DatasetValidator.TEXT_SCHEMA,
        }.get(schema_type, DatasetValidator.TEXT_SCHEMA)
        
        # Collect all fields
        all_fields = set()
        field_type_samples = {}
        missing_field_counts = {field: 0 for field in schema["required_fields"]}
        
        for row in data:
            all_fields.update(row.keys())
            
            # Check required fields
            for field in schema["required_fields"]:
                if field not in row or not row[field]:
                    missing_field_counts[field] += 1
            
            # Sample field types
            for key, value in row.items():
                if key not in field_type_samples:
                    field_type_samples[key] = type(value).__name__
        
        # Validate required fields
        for field, count in missing_field_counts.items():
            if count > 0:
                percent = (count / len(data)) * 100
                if count == len(data):
                    errors.append(f"Required field '{field}' is missing in all rows")
                elif percent > 10:
                    warnings.append(f"Required field '{field}' is missing in {count}/{len(data)} rows ({percent:.1f}%)")
        
        # Calculate text statistics
        avg_input_len = None
        avg_output_len = None
        total_tokens = 0
        
        if schema_type in ["text", "alpaca"]:
            input_lengths = []
            output_lengths = []
            
            for row in data:
                # Input
                input_text = row.get("instruction", "") + " " + row.get("input", "")
                if input_text.strip():
                    input_lengths.append(len(input_text))
                    total_tokens += DatasetValidator.estimate_tokens(input_text)
                
                # Output
                output_text = row.get("output", "")
                if output_text:
                    output_lengths.append(len(output_text))
                    total_tokens += DatasetValidator.estimate_tokens(output_text)
            
            if input_lengths:
                avg_input_len = sum(input_lengths) / len(input_lengths)
            if output_lengths:
                avg_output_len = sum(output_lengths) / len(output_lengths)
        
        # Vision statistics
        has_images = False
        image_count = 0
        image_paths_sample = []
        
        if schema_type == "vision":
            for row in data:
                if "image" in row and row["image"]:
                    has_images = True
                    image_count += 1
                    if len(image_paths_sample) < 5:
                        image_paths_sample.append(row["image"])
        
        # Count total rows (may be more than sample)
        total_rows = len(data)
        if format_type == "jsonl":
            try:
                with open(file_path, 'r') as f:
                    total_rows = sum(1 for line in f if line.strip())
            except Exception:
                pass
        
        # Generate warnings
        if total_rows < 10:
            warnings.append("Dataset has fewer than 10 examples (minimum recommended)")
        elif total_rows < 100:
            warnings.append("Dataset has fewer than 100 examples (recommended for fine-tuning)")
        
        if avg_output_len and avg_output_len < 10:
            warnings.append("Average output length is very short (< 10 characters)")
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            warnings.append("Large dataset file (>100MB) - consider splitting for faster processing")
        
        return DatasetStats(
            total_rows=total_rows,
            format=format_type,
            fields=sorted(list(all_fields)),
            field_types=field_type_samples,
            missing_fields={k: v for k, v in missing_field_counts.items() if v > 0},
            sample_rows=data[:5],  # First 5 rows
            file_size_bytes=file_size,
            validation_errors=errors,
            warnings=warnings,
            avg_input_length=avg_input_len,
            avg_output_length=avg_output_len,
            total_tokens_estimate=total_tokens,
            has_images=has_images,
            image_count=image_count,
            image_paths=image_paths_sample
        )
    
    @staticmethod
    def convert_csv_to_jsonl(csv_path: Path, jsonl_path: Path) -> int:
        """
        Convert CSV to JSONL format.
        
        Args:
            csv_path: Source CSV file
            jsonl_path: Destination JSONL file
            
        Returns:
            Number of rows converted
        """
        count = 0
        with open(csv_path, 'r', encoding='utf-8') as csv_file, \
             open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                jsonl_file.write(json.dumps(row) + '\n')
                count += 1
        return count
    
    @staticmethod
    def convert_jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> int:
        """
        Convert JSONL to CSV format.
        
        Args:
            jsonl_path: Source JSONL file
            csv_path: Destination CSV file
            
        Returns:
            Number of rows converted
        """
        # Read all data to determine fields
        data = []
        all_fields = set()
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    data.append(row)
                    all_fields.update(row.keys())
        
        # Write CSV
        fields = sorted(list(all_fields))
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        
        return len(data)
