#!/usr/bin/env python3
"""
Common fixes for issues found during testing
"""
import sys
import os
import yaml

def fix_import_paths():
    """Fix common import path issues"""
    print("🔧 Checking import paths...")
    
    # Check if src modules can be imported
    sys.path.append('src')
    sys.path.append('.')
    
    # Check directory structure
    if not os.path.exists('src'):
        print("❌ 'src' directory not found!")
        return False
        
    print("📁 Found directories:")
    for item in os.listdir('.'):
        if os.path.isdir(item):
            print(f"  - {item}/")
            
    if os.path.exists('src'):
        print("📁 Found in src/:")
        for item in os.listdir('src'):
            if os.path.isdir(f'src/{item}'):
                print(f"  - src/{item}/")
    
    try:
        # Test basic imports
        import preprocessing
        import models
        import training
        print("✅ All modules importable")
        return True
    except ImportError as e:
        print(f"❌ Import issue: {e}")
        print("💡 Try: export PYTHONPATH=$PWD/src:$PYTHONPATH")
        return False

def fix_config_issues():
    """Fix configuration compatibility issues"""
    print("🔧 Checking configuration...")
    
    try:
        with open('config/test_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
        
        # Validate required keys
        required_keys = ['dataset', 'models', 'training']
        all_good = True
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing config key: {key}")
                all_good = False
            else:
                print(f"✅ Found config key: {key}")
                
        return all_good
                
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def fix_cuda_issues():
    """Check and fix CUDA issues"""
    print("🔧 Checking CUDA...")
    
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("❌ CUDA not available - will use CPU (very slow)")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔧 Checking dependencies...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'sklearn', 
        'onnx', 'onnxruntime', 'yaml', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            
            # Display scikit-learn for clarity but import sklearn
            display_name = 'scikit-learn' if package == 'sklearn' else package
            print(f"✅ {display_name}")
        except ImportError:
            display_name = 'scikit-learn' if package == 'sklearn' else package
            print(f"❌ {display_name}")
            missing.append('scikit-learn' if package == 'sklearn' else package)
    
    if missing:
        print(f"💡 Install missing packages: pip install {' '.join(missing)}")
        return False
    return True

if __name__ == "__main__":
    print("🔧 Running diagnostic checks...")
    
    all_good = True
    all_good &= check_dependencies()
    all_good &= fix_import_paths()
    all_good &= fix_config_issues()
    
    # CUDA is nice to have but not required for basic testing
    cuda_available = fix_cuda_issues()
    if not cuda_available:
        print("⚠️  CUDA not available - tests will run on CPU (slower but functional)")
    
    if all_good:
        print("✅ All essential checks passed!")
        with open('debug_passed.flag', 'w') as f:
            f.write('success')
    else:
        print("❌ Some critical issues found - please fix before proceeding")