"""Health- Check Script for the AI VR Trainer Project.
Run this any time you want to confirm your environment is corrrrectly set up. 

Usage:
    Python scripts/healthcheck.py
    
"""

import sys 
import os 

def check_python_version():
    """We Require Python 3.10.x for this project """
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"[Python vesrion] Found {major}.{minor}.{sys.version_info.micro}")
    if (major, minor) != (3,10):
        print("WARNING: Expected Python 3.10.x You may hit unexpected issues")
        return False
        print(" OK")
        return False
    print("OK")
    return True


def check_virtual_env():
    """ Confirm We are running inside the virtual Environment"""
    in_venv = sys.prefix != sys.base_prefix
    print(f"[Virtual Environment] Activate: {in_venv}")
    print(f"Python executable: {sys.executable}")

    if not in_venv:
        print("WARNING: Not running isnside a virtual environment.")
        print("Did you forget to run '.vemv\\scripts\\activate'?")
        return False
    print("OK")
    return True

def check_imports():
    """Try to import Each core package 
    """

    packages = [
        "numpy",
        "torch",
        "faster_whisper",
        "sentence_transformers",
        "qdrant_client",
        "fastapi",
        "uvicorn",
        "pydantic",
    ]

    print("[Package imports]")
    all_ok = True
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"ok {pkg}")
        except ImportError as e:
            print(f"FAILED {pkg} ({e})")
            all_ok = False
    return all_ok

def check_torch_cpu():
    """confirm torch is the CPU build, since this project runs without GPU."""

    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        print(f"[Torcs build] version = {version}, CUDA available = {cuda_available}")
        if "+cpu" not in version:
            print("WARNING: Expected a '+cpu' build of torch.")
            return False 
        if cuda_available:
            print("WARNING: CUDA reports availabale on a CPU-only machine- unexpected.")
            return False 
        print("OK")
        return True
    except ImportError:
        print("[Torch build] SKIPPED (torch not importable)")
        return False 

def main():
    print("="*50)
    print("AI VR Trainer - Environment Health Check")
    print("="*50)

    results = {
        "Python version": check_python_version(),
        "Virtual environment": check_virtual_env(),
        "Package  imports": check_imports(),
        "Torch CPU build": check_torch_cpu(),
    }

    print("SUMMARY")
    print("="*50)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"[{status}]{name}")
        if not passed:
            all_passed = False

    print("="*50)
    if all_passed:
        print("All checks passed. Environment is ready.")
        sys.exit(0)
    else:
        print("Some check failed. See warnings above")
        sys.exit(1)

if __name__ == "__main__":
    main()
