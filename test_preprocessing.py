#!/usr/bin/env python3
"""
Test preprocessing with your existing code
"""
import sys
import os
sys.path.append('src')

# Test if your preprocessing modules work
def test_imports():
    try:
        from preprocessing import (
            MultiFormatEEGReader,
            SignalQualityAssessor,
            EEGFilter,
            EEGFeatureExtractor,
            SpectrogramGenerator
        )
        print("✅ Preprocessing imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        print("Need to fix import paths or missing dependencies")
        return False

def test_data_preparation():
    """Test the prepare_data.py script with small dataset"""
    try:
        print("🧪 Testing data preparation...")
        result = os.system('python prepare_data.py --config config/test_config.yaml --max-samples 50')
        if result == 0:
            print("✅ Data preparation test completed")
            return True
        else:
            print("❌ Data preparation failed")
            return False
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing preprocessing pipeline...")
    
    success = True
    success &= test_imports()
    success &= test_data_preparation()
    
    if success:
        print("✅ All preprocessing tests passed!")
        with open('preprocessing_test_passed.flag', 'w') as f:
            f.write('success')
    else:
        print("❌ Some preprocessing tests failed!")