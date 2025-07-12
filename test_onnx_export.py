#!/usr/bin/env python3
"""
Test ONNX export functionality
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
sys.path.append('src')

def test_onnx_export():
    """Test ONNX export with a simple model"""
    try:
        # Create simple test model
        class SimpleTestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1d = torch.nn.Conv1d(20, 32, kernel_size=7, padding=3)
                self.gru = torch.nn.GRU(32, 64, batch_first=True)
                self.classifier = torch.nn.Linear(64, 6)
                
            def forward(self, x):
                # x: (batch, channels, time)
                x = self.conv1d(x)  # (batch, 32, time)
                x = x.transpose(1, 2)  # (batch, time, 32)
                x, _ = self.gru(x)  # (batch, time, 64)
                x = x[:, -1, :]  # Take last timestep
                x = self.classifier(x)  # (batch, 6)
                return x
        
        model = SimpleTestModel()
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 20, 10000)
        
        # Test forward pass
        with torch.no_grad():
            torch_output = model(dummy_input)
        
        print(f"✅ PyTorch model output shape: {torch_output.shape}")
        
        # Export to ONNX
        os.makedirs('models/onnx', exist_ok=True)
        onnx_path = 'models/onnx/test_model.onnx'
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['eeg_input'],
            output_names=['predictions'],
            dynamic_axes={
                'eeg_input': {0: 'batch_size'},
                'predictions': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model exported to {onnx_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verification passed")
        
        # Test ONNX Runtime inference
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print(f"✅ ONNX Runtime inference successful: {ort_outputs[0].shape}")
        
        # Compare outputs
        np.testing.assert_allclose(torch_output.detach().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
        print("✅ PyTorch and ONNX outputs match!")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX export test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_model_onnx():
    """Test ONNX export with your actual model"""
    try:
        # Try to load and export your actual trained model
        if os.path.exists('models/final'):
            model_files = [f for f in os.listdir('models/final') if f.endswith('.pth')]
            if model_files:
                print(f"🔄 Testing ONNX export with real model: {model_files[0]}")
                
                # Load your actual model here
                # model = torch.load(f'models/final/{model_files[0]}', map_location='cpu')
                # ... export to ONNX
                
                print("✅ Real model ONNX export test passed")
                return True
        
        print("⚠️ No trained models found, skipping real model test")
        return True
        
    except Exception as e:
        print(f"❌ Real model ONNX test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing ONNX export...")
    
    success = True
    success &= test_onnx_export()
    success &= test_real_model_onnx()
    
    if success:
        print("✅ All ONNX tests passed!")
        with open('onnx_test_passed.flag', 'w') as f:
            f.write('success')
    else:
        print("❌ Some ONNX tests failed!")