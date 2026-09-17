import os
from pathlib import Path

def scan_project():
    # Folders to skip listing (too many files)
    ignore_folders = {'venv', 'build', 'dist', '__pycache__', '.git', 'temp_frames', '_temp_masker_frames'}
    
    # Files absolutely required for the .app build
    required_files = {
        "main.py": "The GUI Launcher (Face)",
        "engine.py": "The SAM 2 Processing Logic (Brain)",
        "ffmpeg": "The video encoding engine",
        "sam2_hiera_s.yaml": "SAM 2 Configuration file",
        "checkpoints/sam2_hiera_small.pt": "AI Model Weights"
    }

    print(f"\n🔍 SCANNING PROJECT: {os.getcwd()}")
    print("="*60)

    found_files = []
    
    # 1. List the Structure (Cleanly)
    for root, dirs, files in os.walk('.'):
        # Filter out ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_folders and not d.startswith('.')]
        
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}📂 {os.path.basename(root) or 'Project Root'}/")
        
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            if f.startswith('.') or f.endswith('.pyc'):
                continue
            print(f"{sub_indent}📄 {f}")
            
            # Track relative path for critical check
            rel_path = os.path.relpath(os.path.join(root, f), '.')
            found_files.append(rel_path)

    print("="*60)
    print("🛡️  CRITICAL ASSET CHECK")
    print("-" * 60)

    all_pass = True
    for file_path, desc in required_files.items():
        if file_path in found_files:
            print(f"✅ FOUND:   {file_path:<35} | {desc}")
        else:
            print(f"❌ MISSING: {file_path:<35} | {desc}")
            all_pass = False

    print("="*60)
    if all_pass:
        print("🚀 STATUS: EVERYTHING IS READY. You can run the PyInstaller command.")
    else:
        print("⚠️  STATUS: STOP. Please fix the missing files before building.")
    print("="*60 + "\n")

if __name__ == "__main__":
    scan_project()