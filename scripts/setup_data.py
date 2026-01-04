#!/usr/bin/env python3
"""
Het-Benchmark Data Setup Script

This script downloads and sets up all required data for Het-Benchmark:
1. Download benchmark input data from GitHub Release
2. Optionally download models from Hugging Face
3. Verify data integrity

Usage:
    # Download benchmark data only
    python setup_data.py --data_only
    
    # Download everything (data + models)
    python setup_data.py --all
    
    # Download specific model category
    python setup_data.py --category LLM
"""

import os
import sys
import json
import argparse
import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

# GitHub Release URL
GITHUB_RELEASE_URL = "https://github.com/wangjingyi34/Het-Benchmark/releases/download/v1.0.0"
BENCHMARK_DATA_FILE = "benchmark_data.tar.gz"
BENCHMARK_DATA_SIZE_MB = 368

# Expected file checksums (SHA256)
CHECKSUMS = {
    "benchmark_data.tar.gz": None,  # Will be computed after upload
}


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    """Download a file with progress indicator."""
    print(f"📥 {desc}...")
    print(f"   URL: {url}")
    print(f"   Destination: {dest_path}")
    
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
        
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r   Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", 
                      end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, report_progress)
        print("\n   ✅ Download complete")
        return True
    except Exception as e:
        print(f"\n   ❌ Download failed: {e}")
        return False


def extract_archive(archive_path: str, dest_dir: str):
    """Extract tar.gz archive."""
    print(f"📦 Extracting {archive_path}...")
    
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        print(f"   ✅ Extracted to {dest_dir}")
        return True
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False


def verify_checksum(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """Verify file checksum."""
    if expected_hash is None:
        print(f"   ⚠️ No checksum available for verification")
        return True
    
    print(f"🔍 Verifying checksum...")
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    actual_hash = sha256.hexdigest()
    if actual_hash == expected_hash:
        print(f"   ✅ Checksum verified")
        return True
    else:
        print(f"   ❌ Checksum mismatch!")
        print(f"   Expected: {expected_hash}")
        print(f"   Actual: {actual_hash}")
        return False


def setup_benchmark_data(dest_dir: str = "./benchmark_data") -> bool:
    """Download and setup benchmark input data."""
    print("\n" + "="*60)
    print("Setting up benchmark input data")
    print("="*60)
    
    archive_path = os.path.join(dest_dir, BENCHMARK_DATA_FILE)
    url = f"{GITHUB_RELEASE_URL}/{BENCHMARK_DATA_FILE}"
    
    # Check if already exists
    manifest_path = os.path.join(dest_dir, "manifest.json")
    if os.path.exists(manifest_path):
        print(f"✅ Benchmark data already exists at {dest_dir}")
        return True
    
    # Download
    if not download_file(url, archive_path, f"Downloading benchmark data (~{BENCHMARK_DATA_SIZE_MB} MB)"):
        return False
    
    # Verify
    verify_checksum(archive_path, CHECKSUMS.get(BENCHMARK_DATA_FILE))
    
    # Extract
    parent_dir = os.path.dirname(dest_dir) or "."
    if not extract_archive(archive_path, parent_dir):
        return False
    
    # Clean up archive
    os.remove(archive_path)
    print(f"   🗑️ Removed archive file")
    
    # Verify extraction
    if os.path.exists(manifest_path):
        print(f"✅ Benchmark data ready at {dest_dir}")
        return True
    else:
        print(f"❌ Extraction verification failed")
        return False


def setup_models(dest_dir: str = "./models_hub", category: Optional[str] = None) -> bool:
    """Download models from Hugging Face."""
    print("\n" + "="*60)
    print("Setting up models from Hugging Face")
    print("="*60)
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    
    # Import model registry
    sys.path.insert(0, str(Path(__file__).parent))
    from download_models import MODEL_REGISTRY, download_model, download_all_models
    
    os.makedirs(dest_dir, exist_ok=True)
    
    if category:
        print(f"Downloading {category} models...")
        download_all_models(dest_dir, category)
    else:
        print("Downloading all models (this may take a while)...")
        download_all_models(dest_dir)
    
    return True


def verify_setup(data_dir: str = "./benchmark_data") -> bool:
    """Verify the setup is complete."""
    print("\n" + "="*60)
    print("Verifying setup")
    print("="*60)
    
    issues = []
    
    # Check benchmark data
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        categories = manifest.get("categories", {})
        print(f"✅ Benchmark data: {len(categories)} categories")
        
        for cat, info in categories.items():
            num_files = info.get("total_files", 0)
            size_mb = info.get("total_size_mb", 0)
            print(f"   - {cat}: {num_files} files ({size_mb:.1f} MB)")
    else:
        issues.append("Benchmark data manifest not found")
        print(f"❌ Benchmark data not found at {data_dir}")
    
    if issues:
        print(f"\n⚠️ Setup incomplete. Issues: {issues}")
        return False
    else:
        print(f"\n✅ Setup complete and verified!")
        return True


def main():
    parser = argparse.ArgumentParser(description="Setup Het-Benchmark data")
    parser.add_argument("--data_dir", type=str, default="./benchmark_data",
                        help="Directory for benchmark input data")
    parser.add_argument("--models_dir", type=str, default="./models_hub",
                        help="Directory for downloaded models")
    parser.add_argument("--data_only", action="store_true",
                        help="Download benchmark data only (no models)")
    parser.add_argument("--models_only", action="store_true",
                        help="Download models only (no benchmark data)")
    parser.add_argument("--all", action="store_true",
                        help="Download everything (data + models)")
    parser.add_argument("--category", type=str, default=None,
                        choices=["LLM", "CV", "NLP", "Audio", "Multimodal"],
                        help="Download models for specific category only")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing setup only")
    
    args = parser.parse_args()
    
    print("🚀 Het-Benchmark Data Setup")
    print("="*60)
    
    if args.verify:
        verify_setup(args.data_dir)
        return
    
    success = True
    
    # Download benchmark data
    if not args.models_only:
        if not setup_benchmark_data(args.data_dir):
            success = False
    
    # Download models
    if args.all or args.models_only:
        if not setup_models(args.models_dir, args.category):
            success = False
    
    # Verify
    verify_setup(args.data_dir)
    
    if success:
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Run benchmark: python scripts/run_benchmark.py")
        print("  2. View results in ./benchmark_results/")
    else:
        print("\n❌ Setup completed with errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
