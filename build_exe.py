
import PyInstaller.__main__
import os
import shutil

def build():
    print("Starting build process...")
    
    # Clean previous builds
    try:
        if os.path.exists("dist"): shutil.rmtree("dist")
        if os.path.exists("build"): shutil.rmtree("build")
    except PermissionError:
        print("\nERROR: Could not delete 'dist' or 'build' folder.")
        print("Please make sure 'ThinkSub2_LoRA_Tool.exe' is CLOSED and try again.")
        return

    # PyInstaller arguments
    args = [
        'gui_trainer.py',  # Main script
        '--name=ThinkSub2_LoRA_Tool',
        '--noconfirm',
        '--onedir',        # Create a directory (portable), not a single exe (too slow to unpack)
        '--noconsole',       # Hide console window (User request)
        
        # Data files and package metadata
        '--add-data=trainer_utils.py:.',
        '--add-data=merge_utils.py:.',
        '--add-data=requirements_gui.txt:.',
        '--collect-all=huggingface_hub',
        '--collect-all=transformers',
        '--collect-all=tokenizers',
        
        # Hidden imports
        '--hidden-import=torch',
        '--hidden-import=torchaudio',
        '--hidden-import=peft',
        '--hidden-import=ctranslate2',
        '--hidden-import=scipy.signal',
        
        # Exclude unnecessary heavyweight stuff if possible (hard with torch)
        # '--exclude-module=tkinter',
    ]
    
    PyInstaller.__main__.run(args)
    
    print("\nBuild complete!")
    print("Executable is in: dist/ThinkSub2_LoRA_Tool/ThinkSub2_LoRA_Tool.exe")
    
    # Post-build: Copy licenses
    if os.path.exists("licenses"):
        print("Copying licenses to dist folder...")
        shutil.copytree("licenses", "dist/ThinkSub2_LoRA_Tool/licenses")
    else:
        print("Warning: 'licenses' folder not found. Run license_gatherer.py first.")

if __name__ == "__main__":
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        os.system("pip install pyinstaller")
        
    build()
