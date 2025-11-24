#!/usr/bin/env python3
"""
Find Ollama Models for llama-cpp-python

This script helps locate Ollama models stored as GGUF blobs and maps them
to their friendly names, making it easy to use them with llama-cpp-python.

Usage:
    python find_ollama_models.py
    python find_ollama_models.py --model mistral:latest
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def get_ollama_models_dir() -> Path:
    """Get the Ollama models directory based on the operating system."""
    home = Path.home()
    
    # Check for custom OLLAMA_MODELS environment variable
    if 'OLLAMA_MODELS' in os.environ:
        return Path(os.environ['OLLAMA_MODELS'])
    
    # Default locations by OS
    return home / ".ollama" / "models"


def parse_manifest(manifest_path: Path) -> Dict:
    """Parse an Ollama manifest file to extract model blob information."""
    try:
        with open(manifest_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        return {}


def find_model_blob(manifest_data: Dict) -> Tuple[str, int]:
    """
    Extract the main model blob hash and size from manifest.
    
    Returns:
        Tuple of (blob_hash, size_in_bytes)
    """
    if not manifest_data:
        return None, 0
    
    # Look for layers - the largest one is typically the model weights
    layers = manifest_data.get('layers', [])
    if not layers:
        return None, 0
    
    # Find the largest layer (this is the model)
    largest_layer = max(layers, key=lambda x: x.get('size', 0))
    
    digest = largest_layer.get('digest', '')
    size = largest_layer.get('size', 0)
    
    # Extract hash from digest (format: sha256:hash)
    if ':' in digest:
        hash_value = digest.split(':', 1)[1]
        return hash_value, size
    
    return None, 0


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def find_all_models() -> List[Dict]:
    """
    Find all Ollama models and their corresponding GGUF blob paths.
    
    Returns:
        List of dicts with model information
    """
    models_dir = get_ollama_models_dir()
    manifests_dir = models_dir / "manifests" / "registry.ollama.ai"
    blobs_dir = models_dir / "blobs"
    
    if not manifests_dir.exists():
        print(f"❌ Manifests directory not found: {manifests_dir}")
        return []
    
    if not blobs_dir.exists():
        print(f"❌ Blobs directory not found: {blobs_dir}")
        return []
    
    models = []
    
    # Walk through manifest directory
    for root, dirs, files in os.walk(manifests_dir):
        for file in files:
            if file.startswith('.'):  # Skip hidden files
                continue
            
            manifest_path = Path(root) / file
            relative_path = manifest_path.relative_to(manifests_dir)
            
            # Construct model name from path
            parts = list(relative_path.parts)
            if len(parts) >= 2:
                if parts[0] == "library":
                    # Official models: library/modelname/tag -> modelname:tag
                    model_name = f"{parts[1]}:{parts[2]}" if len(parts) >= 3 else parts[1]
                else:
                    # Custom models: namespace/modelname/tag -> namespace/modelname:tag
                    model_name = "/".join(parts[:-1]) + ":" + parts[-1]
            else:
                model_name = str(relative_path)
            
            # Parse manifest
            manifest_data = parse_manifest(manifest_path)
            blob_hash, size = find_model_blob(manifest_data)
            
            if blob_hash:
                blob_path = blobs_dir / f"sha256-{blob_hash}"
                
                if blob_path.exists():
                    models.append({
                        'name': model_name,
                        'blob_hash': blob_hash,
                        'blob_path': str(blob_path),
                        'size': size,
                        'size_formatted': format_size(size)
                    })
    
    return sorted(models, key=lambda x: x['name'])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Find Ollama models for use with llama-cpp-python'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Show path for specific model (e.g., mistral:latest)'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export model paths as environment variables'
    )
    
    args = parser.parse_args()
    
    models = find_all_models()
    
    if not models:
        print("No Ollama models found!")
        print("\nMake sure you have models installed with: ollama pull <model>")
        return
    
    if args.model:
        # Show specific model
        matching = [m for m in models if m['name'] == args.model]
        if matching:
            model = matching[0]
            print(f"\n✅ Found: {model['name']}")
            print(f"📦 Size: {model['size_formatted']}")
            print(f"📁 Path: {model['blob_path']}")
            print(f"\n💡 Use with llama-cpp-python:")
            print(f"   export GGUF_MODEL_PATH='{model['blob_path']}'")
            print(f"   python day-1/logit_probabilities.py")
        else:
            print(f"❌ Model '{args.model}' not found!")
            print(f"\nAvailable models:")
            for m in models:
                print(f"  • {m['name']}")
    elif args.export:
        # Export as environment variables
        print("# Export these to use with your scripts:")
        for i, model in enumerate(models, 1):
            var_name = f"OLLAMA_{model['name'].upper().replace(':', '_').replace('/', '_').replace('-', '_')}"
            print(f"export {var_name}='{model['blob_path']}'")
    else:
        # List all models
        print(f"\n🔍 Found {len(models)} Ollama models:\n")
        print(f"{'Model Name':<40} {'Size':<12} {'GGUF Blob Path'}")
        print("=" * 120)
        
        for model in models:
            print(f"{model['name']:<40} {model['size_formatted']:<12} {model['blob_path']}")
        
        print("\n💡 Quick Tips:")
        print("  • Use --model <name> to see the full path for a specific model")
        print("  • Example: python find_ollama_models.py --model mistral:latest")
        print("\n  • Set GGUF_MODEL_PATH to use with logit_probabilities.py:")
        print("    export GGUF_MODEL_PATH='<path_from_above>'")
        print("    python day-1/logit_probabilities.py")
        print("\n  • Recommended small models for testing:")
        print("    - qwen3:0.6b (522 MB) - Smallest, fast")
        print("    - llama3.2:1b (1.3 GB) - Good balance")
        print("    - gemma3:latest (3.3 GB) - High quality")


if __name__ == "__main__":
    main()
