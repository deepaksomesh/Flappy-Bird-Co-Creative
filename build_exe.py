
import os
import subprocess
import sys

def install_pyinstaller():
    print("Checking for PyInstaller...")
    try:
        import PyInstaller
        print("PyInstaller is already installed.")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build():
    print("Building Executable...")
    
    add_data = "assets;assets"
    
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name", "FlappyBirdCoCreative",
        "--add-data", add_data,
        "--hidden-import", "pygame",
        "--hidden-import", "numpy",
        "--hidden-import", "cocreative",
        "--hidden-import", "creative_state",
        "--hidden-import", "world",
        "--hidden-import", "theme",
        "--hidden-import", "sound",
        "--hidden-import", "ui_components",
        "main.py" 
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    
    print("\n" + "="*50)
    print("BUILD SUCCESSFUL!")
    print(f"Executable is located at: {os.path.join(os.getcwd(), 'dist', 'FlappyBirdCoCreative.exe')}")
    print("="*50)

if __name__ == "__main__":
    install_pyinstaller()
    build()
