"""Compare what files are produced by save_pretrained_merged vs merge_vision_lora_adapter."""
import json
from pathlib import Path
import sys

def analyze_model_dir(model_path: str, label: str):
    """Analyze a model directory to understand its structure."""
    print(f"\n{'='*60}")
    print(f"{label}: {model_path}")
    print(f"{'='*60}")
    
    path = Path(model_path)
    if not path.exists():
        print(f"❌ Path does not exist: {model_path}")
        return
    
    # List all files
    print("\n📁 Files in directory:")
    for file in sorted(path.iterdir()):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name:50s} {size_mb:10.2f} MB")
    
    # Check config.json
    config_file = path / "config.json"
    if config_file.exists():
        print("\n📄 config.json content:")
        with open(config_file) as f:
            config = json.load(f)
        
        # Key fields to check
        interesting_fields = [
            "model_type",
            "torch_dtype",
            "_name_or_path",
            "quantization_config",
            "architectures",
        ]
        
        for field in interesting_fields:
            if field in config:
                value = config[field]
                if field == "quantization_config":
                    print(f"  {field}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {field}: {value}")
        
        # Check if text_config exists (for vision models)
        if "text_config" in config:
            print(f"\n  text_config fields:")
            text_config = config["text_config"]
            for field in ["torch_dtype", "quantization_config"]:
                if field in text_config:
                    print(f"    {field}: {text_config[field]}")
    
    # Check for adapter_config.json
    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        print("\n⚠️  Found adapter_config.json - this is a LoRA adapter, not a merged model!")
        with open(adapter_config) as f:
            adapter_cfg = json.load(f)
        print(f"  Base model: {adapter_cfg.get('base_model_name_or_path', 'unknown')}")
        print(f"  LoRA r: {adapter_cfg.get('r', 'unknown')}")
    
    # Check weight file sizes and patterns
    print("\n💾 Weight files:")
    safetensor_files = list(path.glob("*.safetensors"))
    bin_files = list(path.glob("*.bin"))
    
    if safetensor_files:
        print(f"  Found {len(safetensor_files)} .safetensors files")
        total_size_gb = sum(f.stat().st_size for f in safetensor_files) / (1024**3)
        print(f"  Total size: {total_size_gb:.2f} GB")
    
    if bin_files:
        print(f"  Found {len(bin_files)} .bin files")
        total_size_gb = sum(f.stat().st_size for f in bin_files) / (1024**3)
        print(f"  Total size: {total_size_gb:.2f} GB")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_merge_methods.py <model1_path> <model2_path>")
        print()
        print("Example:")
        print("  python compare_merge_methods.py ./models/merged_during_training ./models/merged_after_training")
        sys.exit(1)
    
    analyze_model_dir(sys.argv[1], "Method 1: Merged during training")
    analyze_model_dir(sys.argv[2], "Method 2: Merged after saving LoRA")
    
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print("Look for differences in:")
    print("  1. quantization_config presence")
    print("  2. torch_dtype values")
    print("  3. Weight file sizes (4-bit should be ~4x smaller than 16-bit)")
    print("  4. adapter_config.json presence (should NOT exist in merged models)")
