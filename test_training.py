#!/usr/bin/env python3
"""
Test training pipeline with your existing models
"""
import sys
import torch
import yaml
import os
sys.path.append('src')

def test_model_import():
    """Test if your model classes can be imported"""
    try:
        # Test imports based on your file structure
        from models.resnet1d_gru import ResNet1D_GRU
        from models.efficientnet_spectrogram import EfficientNetSpectrogram
        print("✅ Model imports successful")
        return True
    except Exception as e:
        print(f"❌ Model import error: {e}")
        print("Available files in src/models:")
        if os.path.exists('src/models'):
            for f in os.listdir('src/models'):
                print(f"  - {f}")
        return False

def test_model_creation():
    """Test model creation with config"""
    try:
        with open('config/test_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Test ResNet1D-GRU creation
        from models.resnet1d_gru import ResNet1D_GRU
        model = ResNet1D_GRU(config)
        print(f"✅ ResNet1D-GRU model created: {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        dummy_input = torch.randn(2, 19, 10000)  # batch=2, channels=19, time=10000
        if torch.cuda.is_available():
            model = model.cuda()
            dummy_input = dummy_input.cuda()
            
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Forward pass successful: logits shape {output['logits'].shape}")
        
        return True, model
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_training_components():
    """Test training components"""
    try:
        # Import your trainer
        from training.trainer import HMSTrainer
        
        print("✅ Trainer import successful")
        return True
    except Exception as e:
        print(f"❌ Training component test failed: {e}")
        print("Available files in src/training:")
        if os.path.exists('src/training'):
            for f in os.listdir('src/training'):
                print(f"  - {f}")
        return False

if __name__ == "__main__":
    print("🧪 Testing model components...")
    
    success = True
    success &= test_model_import()
    
    if success:
        model_success, model = test_model_creation()
        success &= model_success
        
    success &= test_training_components()
    
    if success:
        print("✅ All model tests passed!")
        with open('model_test_passed.flag', 'w') as f:
            f.write('success')
    else:
        print("❌ Some model tests failed!")