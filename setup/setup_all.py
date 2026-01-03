#!/usr/bin/env python3
"""
YOLO-SAM Toolkit - Hardware Detection and PyTorch Installation
Detects your GPU type and installs the correct PyTorch version
"""
import subprocess
import platform
import os
from pathlib import Path

def run_command(cmd, cwd=None):
    """Executes shell commands and streams output to the console in real-time"""
    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, cwd=cwd
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        return process.returncode == 0
    except Exception as e:
        print(f"Error executing command: {e}")
        return False

def detect_hardware():
    """Detects hardware type (NVIDIA, AMD, or CPU) across Windows and Linux"""
    os_name = platform.system()
    print(f"--- Detected Operating System: {os_name} ---")
    
    # 1. Windows Detection Logic
    if os_name == "Windows":
        # Use wmic to find GPU names
        status, stdout = subprocess.getstatusoutput("wmic path win32_VideoController get name")
        if status == 0 and "NVIDIA" in stdout.upper():
            print("[DETECTED] NVIDIA GPU (Windows)")
            return "nvidia"
        elif status == 0 and "AMD" in stdout.upper():
            print("[DETECTED] AMD GPU (Windows) -> Note: Using CPU mode for stability")
            return "cpu"
            
    # 2. Linux Detection Logic
    elif os_name == "Linux":
        # Check for NVIDIA drivers/SMI
        status, _ = subprocess.getstatusoutput("nvidia-smi")
        if status == 0:
            print("[DETECTED] NVIDIA GPU (Linux)")
            return "nvidia"
        
        # Check for AMD ROCm
        status, _ = subprocess.getstatusoutput("rocm-smi")
        if status == 0:
            print("[DETECTED] AMD GPU (Linux)")
            return "amd"

    print("[INFO] No compatible GPU/Drivers found. Falling back to CPU mode.")
    return "cpu"

def get_install_commands(hw_type):
    """Returns the appropriate installation strings based on hardware"""
    # PyTorch with CUDA 12.1 for NVIDIA, ROCm 6.1 for AMD, or standard CPU
    targets = {
        "nvidia": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "amd": "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1",
        "cpu": "torch torchvision torchaudio"
    }
    
    py_cmd = f"pip install {targets[hw_type]}"
    yolo_cmd = "pip install ultralytics"
    
    return py_cmd, yolo_cmd

def verify_install():
    """Verifies that the core components are correctly installed and accessible"""
    print("\n--- Verifying Installation ---")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
        
        gpu_ready = torch.cuda.is_available()
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] GPU available for PyTorch: {'YES' if gpu_ready else 'NO'}")
        print(f"[OK] YOLOv8 (Ultralytics) version: {ultralytics.__version__}")
        return True
    except ImportError as e:
        print(f"[ERROR] Required library not found: {e}")
        return False

def download_sam2_models_only(models_to_download):
    """Downloads SAM 2 models without installing SAM 2 (assumes it's already installed)"""
    print("\n--- Downloading SAM 2 Models Only ---")
    
    # Check if SAM 2 directory exists
    if not os.path.exists("segment-anything-2"):
        print("[WARNING] SAM 2 directory not found at 'segment-anything-2'")
        create_dir = input("Create directory structure? (y/n): ").strip().lower()
        if create_dir != 'y':
            print("[INFO] Skipping model download. Please ensure SAM 2 is installed.")
            return False
    
    # Create checkpoints directory
    ckpt_dir = Path("segment-anything-2/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary with all official SAM 2.1 models
    available_models = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    }
    
    # Download selected models
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        run_command("pip install requests")
        import requests
    
    for model_name in models_to_download:
        if model_name not in available_models:
            print(f"[WARNING] Unknown model: {model_name}, skipping...")
            continue
        
        url = available_models[model_name]
        dest = ckpt_dir / f"sam2.1_hiera_{model_name}.pt"
        
        if dest.exists():
            print(f"[INFO] Model '{model_name}' already downloaded, skipping...")
            continue
        
        print(f"📥 Downloading SAM 2.1 {model_name.upper()}... (this may take a while)")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="")
            
            print(f"\n✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"\n[ERROR] Failed to download {model_name}: {e}")
            return False
    
    print("\n✅ SAM 2 models download complete!")
    return True

def setup_sam2(models_to_download):
    """Clones SAM 2 repository and downloads selected models"""
    print("\n--- Setting up SAM 2 ---")
    
    # Clone repository if it doesn't exist
    if not os.path.exists("segment-anything-2"):
        print("Cloning SAM 2 repository...")
        if not run_command("git clone https://github.com/facebookresearch/segment-anything-2.git"):
            print("[ERROR] Failed to clone SAM 2 repository")
            return False
    else:
        print("[INFO] SAM 2 repository already exists")
    
    # Install in editable mode
    print("Installing SAM 2...")
    if not run_command("pip install -e .", cwd="segment-anything-2"):
        print("[ERROR] Failed to install SAM 2")
        return False
    
    # Create checkpoints directory
    ckpt_dir = Path("segment-anything-2/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary with all official SAM 2.1 models
    available_models = {
        "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "large": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    }
    
    # Download selected models
    try:
        import requests
    except ImportError:
        print("Installing requests library...")
        run_command("pip install requests")
        import requests
    
    for model_name in models_to_download:
        if model_name not in available_models:
            print(f"[WARNING] Unknown model: {model_name}, skipping...")
            continue
        
        url = available_models[model_name]
        dest = ckpt_dir / f"sam2.1_hiera_{model_name}.pt"
        
        if dest.exists():
            print(f"[INFO] Model '{model_name}' already downloaded, skipping...")
            continue
        
        print(f"📥 Downloading SAM 2.1 {model_name.upper()}... (this may take a while)")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="")
            
            print(f"\n✅ {model_name} downloaded successfully")
        except Exception as e:
            print(f"\n[ERROR] Failed to download {model_name}: {e}")
            return False
    
    print("\n✅ SAM 2 setup complete!")
    return True

def get_sam2_selection():
    """Prompts user to select SAM2 models to download"""
    print("\n" + "="*60)
    print("SAM 2 MODEL SELECTION")
    print("="*60)
    print("\nAvailable models:")
    print("  - tiny       (Fastest, least accurate)")
    print("  - small      (Fast, good accuracy)")
    print("  - base_plus  (Balanced)")
    print("  - large      (Slowest, best accuracy)")
    print("\nYou can select multiple models separated by commas")
    print("Example: tiny,large  or just: large")
    print("Type 'skip' to not download any models")
    
    while True:
        choice = input("\nWhich model(s) do you want to download? ").strip().lower()
        
        if choice == 'skip':
            return []
        
        if not choice:
            print("[ERROR] Please enter at least one model name or 'skip'")
            continue
        
        # Parse selection
        selected = [m.strip() for m in choice.split(',')]
        valid_models = {"tiny", "small", "base_plus", "large"}
        
        # Validate all selections
        invalid = [m for m in selected if m not in valid_models]
        if invalid:
            print(f"[ERROR] Invalid model(s): {', '.join(invalid)}")
            print(f"Valid options: {', '.join(valid_models)}")
            continue
        
        # Show selection and confirm
        print(f"\nYou selected: {', '.join(selected)}")
        confirm = input("Sure? (y/n): ").strip().lower()
        
        if confirm == 'y':
            return selected
        else:
            print("Let's try again...")

def main():
    print("="*60)
    print("MODULAR INSTALLER: YOLOv8 + SAM 2")
    print("="*60)

    # 1. Detection Phase
    hw = detect_hardware()
    py_cmd, yolo_cmd = get_install_commands(hw)

    # 2. Ask about PyTorch/YOLO installation
    print(f"\nDetected hardware: {hw.upper()}")
    print(f"PyTorch will be optimized for: {hw.upper()}")
    print(f"This includes: PyTorch + TorchVision + TorchAudio + YOLOv8")
    
    install_pytorch = input("\nDo you want to install PyTorch and YOLOv8? (y/n): ").strip().lower()
    
    # 3. Ask about SAM 2 installation
    print("\n" + "="*60)
    print("SAM 2 INSTALLATION OPTIONS")
    print("="*60)
    print("1. Full installation (clone repository + install SAM 2 + download models)")
    print("2. Models only (download models, skip SAM 2 installation)")
    print("3. Skip SAM 2 completely")
    
    sam2_option = input("\nSelect option (1/2/3): ").strip()
    
    sam2_models = []
    install_full_sam2 = False
    download_models_only = False
    
    if sam2_option == '1':
        install_full_sam2 = True
        sam2_models = get_sam2_selection()
    elif sam2_option == '2':
        download_models_only = True
        sam2_models = get_sam2_selection()
    elif sam2_option == '3':
        print("SAM 2 installation skipped.")
    else:
        print("[WARNING] Invalid option. Skipping SAM 2 installation.")
    
    # 4. Check if user wants to install anything
    if install_pytorch != 'y' and not sam2_models and not install_full_sam2:
        print("\nNo components selected for installation. Exiting.")
        return
    
    # 5. Show installation summary
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    if install_pytorch == 'y':
        print(f"✓ PyTorch for {hw.upper()}")
        print(f"✓ YOLOv8 (Ultralytics)")
    if install_full_sam2:
        print(f"✓ SAM 2 (full installation)")
    if download_models_only:
        print(f"✓ SAM 2 models only (no installation)")
    if sam2_models:
        print(f"✓ Models to download: {', '.join(sam2_models)}")
    
    confirm = input("\nProceed with installation? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Installation cancelled by user.")
        return

    # 6. Installation Phase - PyTorch and YOLO
    if install_pytorch == 'y':
        print("\n--- Installing PyTorch (This may take a few minutes) ---")
        if not run_command(py_cmd):
            print("\n[!] Failed to install PyTorch.")
            return
        
        print("\n--- Installing YOLOv8 ---")
        if not run_command(yolo_cmd):
            print("\n[!] Failed to install YOLOv8.")
            return
        
        # Verify PyTorch/YOLO installation
        if not verify_install():
            print("\n[!] Installation completed but verification failed.")
            return
    
    # 7. Installation Phase - SAM 2 (full installation)
    if install_full_sam2 and sam2_models:
        if not setup_sam2(sam2_models):
            print("\n[!] SAM 2 installation failed.")
            return
    
    # 8. Download models only (SAM 2 already installed)
    if download_models_only and sam2_models:
        if not download_sam2_models_only(sam2_models):
            print("\n[!] SAM 2 models download failed.")
            return
    
    # 9. Success message
    print("\n" + "="*60)
    print("SUCCESS! Your environment is ready.")
    if install_pytorch == 'y':
        print("✅ PyTorch and YOLOv8 installed")
    if install_full_sam2:
        print(f"✅ SAM 2 installed with models: {', '.join(sam2_models) if sam2_models else 'none'}")
    if download_models_only and sam2_models:
        print(f"✅ SAM 2 models downloaded: {', '.join(sam2_models)}")
    print("Next step: Start building your pipeline!")
    print("="*60)

if __name__ == "__main__":
    main()
    