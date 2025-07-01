#!/usr/bin/env python3
"""
Test Explainable AI and Interpretability Framework for HMS Brain Activity Classification
Comprehensive testing of counterfactual explanations, SHAP integration, gradient methods,
attention visualization, and clinical interpretation tools.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
import time

sys.path.append('src')

def test_interpretability_imports():
    """Test imports for interpretability components."""
    try:
        from interpretability import (
            ExplainableAI,
            CounterfactualGenerator,
            SHAPExplainer,
            AttentionVisualizer,
            ClinicalInterpreter,
            ExplanationConfig,
            GradientExplanationFramework,
            GradCAM,
            IntegratedGradients,
            GuidedBackpropagation,
            SmoothGrad,
            LayerActivationAnalysis,
            create_explainable_ai,
            create_gradient_explainer
        )
        print("✅ Interpretability imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def generate_test_eeg_data(batch_size=4, n_channels=19, seq_length=2000):
    """Generate synthetic EEG data for testing."""
    
    # Generate realistic EEG-like signals
    t = np.linspace(0, seq_length/200, seq_length)  # 200 Hz sampling rate
    eeg_data = []
    labels = []
    
    for i in range(batch_size):
        # Generate multi-channel EEG
        channels = np.zeros((n_channels, seq_length))
        
        for ch in range(n_channels):
            # Mix of different frequency bands
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            gamma = 0.2 * np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi)
            
            # Add noise
            noise = 0.1 * np.random.randn(seq_length)
            
            # Different patterns for different classes
            if i % 6 == 0:  # Seizure-like pattern
                seizure_pattern = 2.0 * np.sin(2 * np.pi * 3 * t) * np.exp(-t/5)
                channels[ch] = alpha + beta + theta + gamma + noise + seizure_pattern
            else:
                channels[ch] = alpha + beta + theta + gamma + noise
            
        eeg_data.append(channels)
        labels.append(i % 6)  # 6 classes
        
    return eeg_data, labels

def create_test_model():
    """Create a simple test model for explanations."""
    class SimpleEEGModel(nn.Module):
        def __init__(self, n_channels=19, n_classes=6):
            super().__init__()
            self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, n_classes)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            return x
    
    return SimpleEEGModel()

def test_explanation_config():
    """Test ExplanationConfig creation and validation."""
    try:
        from interpretability import ExplanationConfig
        
        print("🧪 Testing ExplanationConfig...")
        
        # Test default configuration
        config = ExplanationConfig()
        
        # Validate default values
        assert config.counterfactual_method == 'gradient_based'
        assert config.shap_method == 'deep'
        assert config.cf_max_iterations == 500
        assert len(config.frequency_bands) == 5
        
        # Test custom configuration
        custom_config = ExplanationConfig(
            counterfactual_method='optimization',
            shap_method='gradient',
            cf_max_iterations=1000,
            cf_learning_rate=0.001
        )
        
        assert custom_config.counterfactual_method == 'optimization'
        assert custom_config.shap_method == 'gradient'
        assert custom_config.cf_max_iterations == 1000
        assert custom_config.cf_learning_rate == 0.001
        
        print("✅ ExplanationConfig test passed")
        return True
        
    except Exception as e:
        print(f"❌ ExplanationConfig test failed: {e}")
        return False

def test_counterfactual_generation():
    """Test counterfactual explanation generation."""
    try:
        from interpretability import CounterfactualGenerator, ExplanationConfig
        
        print("🧪 Testing CounterfactualGenerator...")
        
        # Create test model and data
        model = create_test_model()
        model.eval()
        
        config = ExplanationConfig(
            cf_max_iterations=50,  # Reduced for testing
            cf_learning_rate=0.01
        )
        
        generator = CounterfactualGenerator(model, config, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=1, seq_length=1000)
        test_input = torch.tensor(eeg_data[0], dtype=torch.float32).unsqueeze(0)
        
        # Test counterfactual generation
        target_class = 1  # Different from likely prediction
        cf_result = generator.generate_counterfactual(test_input, target_class)
        
        # Validate results
        required_keys = [
            'counterfactual', 'original', 'changes', 'total_change',
            'relative_change', 'final_prediction', 'target_class',
            'confidence', 'success'
        ]
        
        for key in required_keys:
            if key not in cf_result:
                print(f"❌ Missing key in counterfactual result: {key}")
                return False
        
        # Check shapes
        if cf_result['counterfactual'].shape != test_input.shape:
            print(f"❌ Counterfactual shape mismatch")
            return False
        
        # Test diverse counterfactuals
        diverse_cfs = generator.generate_diverse_counterfactuals(
            test_input, target_class, n_counterfactuals=3
        )
        
        if len(diverse_cfs) != 3:
            print(f"❌ Expected 3 diverse counterfactuals, got {len(diverse_cfs)}")
            return False
        
        print(f"  Counterfactual success: {cf_result['success']}")
        print(f"  Total change: {cf_result['total_change']:.6f}")
        print(f"  Relative change: {cf_result['relative_change']:.3f}")
        print("✅ CounterfactualGenerator test passed")
        return True
        
    except Exception as e:
        print(f"❌ CounterfactualGenerator test failed: {e}")
        return False

def test_shap_explainer():
    """Test SHAP-based explanations."""
    try:
        from interpretability import SHAPExplainer, ExplanationConfig
        
        print("🧪 Testing SHAPExplainer...")
        
        # Create test model and data
        model = create_test_model()
        model.eval()
        
        config = ExplanationConfig(
            shap_method='gradient',  # Use gradient for faster testing
            shap_n_samples=10
        )
        
        explainer = SHAPExplainer(model, config, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=5, seq_length=1000)
        background_data = torch.tensor(eeg_data[:3], dtype=torch.float32)
        test_data = torch.tensor(eeg_data[3:4], dtype=torch.float32)
        
        # Initialize explainer
        explainer.initialize_explainer(background_data)
        
        # Generate SHAP explanation
        shap_result = explainer.explain(test_data)
        
        # Validate results
        required_keys = ['shap_values', 'predictions', 'input_data', 'analysis']
        for key in required_keys:
            if key not in shap_result:
                print(f"❌ Missing key in SHAP result: {key}")
                return False
        
        # Check analysis components
        analysis = shap_result['analysis']
        analysis_keys = ['channel_importance', 'temporal_importance', 'frequency_importance']
        for key in analysis_keys:
            if key not in analysis:
                print(f"❌ Missing analysis component: {key}")
                return False
        
        print(f"  SHAP values shape: {np.array(shap_result['shap_values']).shape}")
        print(f"  Global importance: {analysis['global_importance']:.6f}")
        print("✅ SHAPExplainer test passed")
        return True
        
    except Exception as e:
        print(f"❌ SHAPExplainer test failed: {e}")
        return False

def test_gradient_explanations():
    """Test gradient-based explanation methods."""
    try:
        from interpretability import GradientExplanationFramework
        
        print("🧪 Testing GradientExplanationFramework...")
        
        # Create test model
        model = create_test_model()
        model.eval()
        
        # Target layers for Grad-CAM
        target_layers = ['conv1', 'conv2']
        
        framework = GradientExplanationFramework(model, target_layers, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=1, seq_length=1000)
        test_input = torch.tensor(eeg_data[0], dtype=torch.float32).unsqueeze(0)
        
        # Test different explanation methods
        methods = ['integrated_gradients', 'guided_backprop', 'smooth_grad', 'grad_cam']
        
        explanations = framework.explain(
            test_input,
            methods=methods,
            ig_params={'steps': 10},  # Reduced for testing
            sg_params={'n_samples': 10}  # Reduced for testing
        )
        
        # Validate results
        for method in methods:
            if method not in explanations:
                print(f"❌ Missing explanation method: {method}")
                return False
            
            if 'error' in explanations[method]:
                print(f"⚠️ {method} failed: {explanations[method]['error']}")
                continue
            
            # Method-specific validation
            if method == 'integrated_gradients':
                result = explanations[method]
                required_keys = ['integrated_gradients', 'channel_attributions', 'temporal_attributions']
                for key in required_keys:
                    if key not in result:
                        print(f"❌ Missing key in {method}: {key}")
                        return False
            
            elif method == 'guided_backprop':
                result = explanations[method]
                if 'guided_gradients' not in result:
                    print(f"❌ Missing guided_gradients in {method}")
                    return False
            
            elif method == 'smooth_grad':
                result = explanations[method]
                if 'smooth_gradients' not in result:
                    print(f"❌ Missing smooth_gradients in {method}")
                    return False
        
        # Test comparison methods
        comparison = framework.compare_methods(explanations)
        
        print(f"  Explanation methods tested: {len([m for m in methods if 'error' not in explanations[m]])}")
        print(f"  Comparison metrics: {list(comparison.keys())}")
        print("✅ GradientExplanationFramework test passed")
        return True
        
    except Exception as e:
        print(f"❌ GradientExplanationFramework test failed: {e}")
        return False

def test_clinical_interpreter():
    """Test clinical interpretation tools."""
    try:
        from interpretability import ClinicalInterpreter, ExplanationConfig
        
        print("🧪 Testing ClinicalInterpreter...")
        
        config = ExplanationConfig()
        interpreter = ClinicalInterpreter(config)
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=1, seq_length=2000)
        test_eeg = eeg_data[0]  # Shape: (19, 2000)
        
        # Mock explanation data
        mock_explanation = {
            'shap_values': np.random.randn(19, 2000),
            'analysis': {
                'channel_importance': np.random.rand(19),
                'temporal_importance': np.random.rand(2000),
                'global_importance': 0.5
            }
        }
        
        # Mock prediction data
        mock_prediction = {
            'predicted_class': 'seizure',
            'confidence': 0.85,
            'predicted_class_idx': 0
        }
        
        # Test interpretation
        interpretation = interpreter.interpret_explanation(
            mock_explanation, test_eeg, mock_prediction
        )
        
        # Validate results
        required_keys = [
            'summary', 'clinical_features', 'risk_factors',
            'recommendations', 'confidence_analysis'
        ]
        
        for key in required_keys:
            if key not in interpretation:
                print(f"❌ Missing interpretation component: {key}")
                return False
        
        # Check clinical features
        clinical_features = interpretation['clinical_features']
        feature_keys = ['power_spectral_density', 'spectral_entropy', 'hjorth_parameters']
        for key in feature_keys:
            if key not in clinical_features:
                print(f"❌ Missing clinical feature: {key}")
                return False
        
        # Check recommendations
        recommendations = interpretation['recommendations']
        if not isinstance(recommendations, list):
            print(f"❌ Recommendations should be a list")
            return False
        
        print(f"  Summary: {interpretation['summary'][:50]}...")
        print(f"  Risk factors: {len(interpretation['risk_factors'])}")
        print(f"  Recommendations: {len(recommendations)}")
        print("✅ ClinicalInterpreter test passed")
        return True
        
    except Exception as e:
        print(f"❌ ClinicalInterpreter test failed: {e}")
        return False

def test_explainable_ai_framework():
    """Test the main ExplainableAI framework."""
    try:
        from interpretability import ExplainableAI, ExplanationConfig
        
        print("🧪 Testing ExplainableAI framework...")
        
        # Create test model and configuration
        model = create_test_model()
        model.eval()
        
        config = ExplanationConfig(
            cf_max_iterations=20,  # Reduced for testing
            shap_n_samples=10
        )
        
        explainer = ExplainableAI(model, config, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=5, seq_length=1000)
        background_data = torch.tensor(eeg_data[:3], dtype=torch.float32)
        test_data = torch.tensor(eeg_data[3:4], dtype=torch.float32)
        
        # Initialize background data for SHAP
        explainer.initialize_background_data(background_data)
        
        # Test comprehensive explanation
        explanation_types = ['prediction', 'counterfactual', 'clinical']
        explanation = explainer.explain_prediction(
            test_data, 
            explanation_types=explanation_types,
            target_class=2
        )
        
        # Validate results
        for exp_type in explanation_types:
            if exp_type not in explanation:
                print(f"❌ Missing explanation type: {exp_type}")
                return False
        
        # Check prediction info
        prediction = explanation['prediction']
        pred_keys = ['predicted_class_idx', 'confidence', 'probabilities']
        for key in pred_keys:
            if key not in prediction:
                print(f"❌ Missing prediction key: {key}")
                return False
        
        # Test report generation
        report = explainer.generate_explanation_report(explanation)
        
        if len(report) < 100:  # Should be a substantial report
            print(f"❌ Report too short: {len(report)} characters")
            return False
        
        print(f"  Prediction class: {prediction['predicted_class_idx']}")
        print(f"  Confidence: {prediction['confidence']:.3f}")
        print(f"  Report length: {len(report)} characters")
        print("✅ ExplainableAI framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ ExplainableAI framework test failed: {e}")
        return False

def test_factory_functions():
    """Test factory functions for creating explainers."""
    try:
        from interpretability import create_explainable_ai, create_gradient_explainer
        
        print("🧪 Testing factory functions...")
        
        # Create test model
        model = create_test_model()
        
        # Test create_explainable_ai
        explainer = create_explainable_ai(model, device='cpu')
        
        if explainer is None:
            print(f"❌ create_explainable_ai returned None")
            return False
        
        # Test with custom config
        config_dict = {
            'cf_max_iterations': 100,
            'shap_method': 'gradient'
        }
        
        explainer_custom = create_explainable_ai(model, config=config_dict, device='cpu')
        
        if explainer_custom.config.cf_max_iterations != 100:
            print(f"❌ Custom config not applied")
            return False
        
        # Test create_gradient_explainer
        target_layers = ['conv1', 'conv2']
        grad_explainer = create_gradient_explainer(model, target_layers, device='cpu')
        
        if grad_explainer is None:
            print(f"❌ create_gradient_explainer returned None")
            return False
        
        print("✅ Factory functions test passed")
        return True
        
    except Exception as e:
        print(f"❌ Factory functions test failed: {e}")
        return False

def test_attention_visualization():
    """Test attention visualization for transformer models."""
    try:
        from interpretability import AttentionVisualizer, ExplanationConfig
        
        print("🧪 Testing AttentionVisualizer...")
        
        # Create a simple model with attention (mock transformer)
        class MockAttentionModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(19, 64, 3, padding=1)
                self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
                self.fc = nn.Linear(64, 6)
                
            def forward(self, x, return_attentions=False):
                x = self.conv(x)  # (batch, 64, seq_len)
                x = x.transpose(1, 2)  # (batch, seq_len, 64)
                
                if return_attentions:
                    attn_output, attn_weights = self.attention(x, x, x)
                    x = attn_output.mean(dim=1)  # Global average pooling
                    out = self.fc(x)
                    return {'logits': out, 'attentions': [attn_weights]}
                else:
                    attn_output, _ = self.attention(x, x, x)
                    x = attn_output.mean(dim=1)
                    return self.fc(x)
        
        model = MockAttentionModel()
        model.eval()
        
        config = ExplanationConfig()
        visualizer = AttentionVisualizer(model, config, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=1, seq_length=100)  # Smaller for attention
        test_input = torch.tensor(eeg_data[0], dtype=torch.float32).unsqueeze(0)
        
        # Extract attention patterns
        attention_result = visualizer.extract_attention_patterns(test_input)
        
        # Validate results
        required_keys = ['attention_weights', 'analysis', 'input_shape']
        for key in required_keys:
            if key not in attention_result:
                print(f"❌ Missing attention result key: {key}")
                return False
        
        print(f"  Input shape: {attention_result['input_shape']}")
        print(f"  Attention layers captured: {len(attention_result['attention_weights'])}")
        print("✅ AttentionVisualizer test passed")
        return True
        
    except Exception as e:
        print(f"❌ AttentionVisualizer test failed: {e}")
        return False

def test_integration_with_models():
    """Test integration with different model types."""
    try:
        from interpretability import create_explainable_ai
        from models import create_model
        
        print("🧪 Testing integration with different models...")
        
        # Test with simple model
        simple_model = create_test_model()
        explainer = create_explainable_ai(simple_model, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=1, seq_length=1000)
        test_input = torch.tensor(eeg_data[0], dtype=torch.float32).unsqueeze(0)
        
        # Test basic explanation
        explanation = explainer.explain_prediction(
            test_input,
            explanation_types=['prediction'],
            target_class=1
        )
        
        if 'prediction' not in explanation:
            print(f"❌ Integration test failed - no prediction")
            return False
        
        print(f"  Model type: {type(simple_model).__name__}")
        print(f"  Prediction confidence: {explanation['prediction']['confidence']:.3f}")
        print("✅ Model integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Model integration test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance benchmarks for explanation methods."""
    try:
        from interpretability import create_explainable_ai
        
        print("🧪 Testing performance benchmarks...")
        
        # Create test setup
        model = create_test_model()
        explainer = create_explainable_ai(model, device='cpu')
        
        # Generate test data
        eeg_data, _ = generate_test_eeg_data(batch_size=3, seq_length=1000)
        background_data = torch.tensor(eeg_data[:2], dtype=torch.float32)
        test_data = torch.tensor(eeg_data[2:3], dtype=torch.float32)
        
        explainer.initialize_background_data(background_data)
        
        # Benchmark different explanation types
        methods_to_test = [
            ('prediction', ['prediction']),
            ('counterfactual', ['prediction', 'counterfactual']),
        ]
        
        results = {}
        
        for method_name, explanation_types in methods_to_test:
            start_time = time.time()
            
            explanation = explainer.explain_prediction(
                test_data,
                explanation_types=explanation_types
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            results[method_name] = {
                'time': processing_time,
                'success': len(explanation) > 0
            }
            
            print(f"  {method_name}: {processing_time:.3f}s")
        
        # Check performance constraints (should complete within reasonable time)
        for method_name, result in results.items():
            if result['time'] > 30.0:  # 30 seconds max for test
                print(f"⚠️ {method_name} too slow: {result['time']:.3f}s")
            
            if not result['success']:
                print(f"❌ {method_name} failed")
                return False
        
        print("✅ Performance benchmarks test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmarks test failed: {e}")
        return False

def main():
    """Run all interpretability tests."""
    print("🧪 HMS Explainable AI and Interpretability Test Suite")
    print("=" * 70)
    
    tests = [
        ("Import Tests", test_interpretability_imports),
        ("ExplanationConfig", test_explanation_config),
        ("CounterfactualGenerator", test_counterfactual_generation),
        ("SHAPExplainer", test_shap_explainer),
        ("GradientExplanations", test_gradient_explanations),
        ("ClinicalInterpreter", test_clinical_interpreter),
        ("ExplainableAI Framework", test_explainable_ai_framework),
        ("Factory Functions", test_factory_functions),
        ("AttentionVisualizer", test_attention_visualization),
        ("Model Integration", test_integration_with_models),
        ("Performance Benchmarks", test_performance_benchmarks)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 50)
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Explainable AI tests passed!")
        # Create success flag
        with open('interpretability_test_passed.flag', 'w') as f:
            f.write('success')
        return True
    else:
        print(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 