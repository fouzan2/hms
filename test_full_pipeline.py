#!/usr/bin/env python3
"""
Full pipeline integration test
"""
import os
import sys
import time
import yaml

def run_full_test():
    """Run the complete pipeline test"""
    
    print("🚀 Starting full pipeline test...")
    start_time = time.time()
    
    # Check prerequisites
    required_flags = ['debug_passed.flag']
    for flag in required_flags:
        if not os.path.exists(flag):
            print(f"❌ Missing prerequisite: {flag}")
            print("Please run debug_fixes.py first")
            return False
    
    # Step 1: Data preparation (should be quick with mock data)
    print("\n📊 Step 1: Data Preparation")
    if not os.path.exists('data/raw/train.csv'):
        print("Running test data preparation...")
        if os.system('python test_data_prep.py') != 0:
            print("❌ Test data preparation failed")
            return False
    
    # Test actual preprocessing
    if os.system('python prepare_data.py --config config/test_config.yaml --max-samples 50') != 0:
        print("❌ Data preparation failed")
        return False
    print("✅ Data preparation completed")
    
    # Step 2: Quick training test
    print("\n🏋️ Step 2: Training Test")
    try:
        # Test if we can import and run trainer
        sys.path.append('src')
        
        print("🔄 Testing training components...")
        
        # Create a minimal training test
        minimal_train_script = """
import torch
import sys
sys.path.append('src')

try:
    from models.resnet1d_gru import ResNet1D_GRU
    import yaml
    
    with open('config/test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = ResNet1D_GRU(config)
    print("✅ Model created successfully")
    
    # Save a dummy trained model for ONNX testing
    import os
    os.makedirs('models/final', exist_ok=True)
    torch.save(model.state_dict(), 'models/final/test_model.pth')
    print("✅ Model saved")
    
except Exception as e:
    print(f"❌ Training test failed: {e}")
    import traceback
    traceback.print_exc()
"""
        
        with open('temp_train_test.py', 'w') as f:
            f.write(minimal_train_script)
        
        if os.system('python temp_train_test.py') != 0:
            print("❌ Training test failed")
            return False
            
        os.remove('temp_train_test.py')
        print("✅ Training test completed")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    # Step 3: ONNX export test
    print("\n📦 Step 3: ONNX Export Test")
    if os.system('python test_onnx_export.py') != 0:
        print("❌ ONNX export failed")
        return False
    print("✅ ONNX export completed")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Full pipeline test completed in {total_time:.1f} seconds")
    
    # Validate outputs
    expected_files = [
        'data/processed/processing_summary.csv',
        'models/onnx/test_model.onnx',
        'models/final/test_model.pth'
    ]
    
    all_files_found = True
    for file_path in expected_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_files_found = False
    
    if all_files_found:
        with open('test_passed.flag', 'w') as f:
            f.write('success')
        print("✅ All tests passed! Ready for Novita AI deployment.")
        return True
    else:
        print("❌ Some output files missing")
        return False

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)