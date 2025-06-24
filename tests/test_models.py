"""Unit tests for model architectures."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

from src.models import (
    ResNet1D_GRU, 
    EfficientNetSpectrogram, 
    HMSEnsembleModel,
    MetaLearner
)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestResNet1DGRU:
    """Test ResNet1D-GRU model."""
    
    def test_init(self):
        """Test model initialization."""
        model = ResNet1D_GRU(
            num_channels=19,
            num_classes=6,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        
        assert model.num_channels == 19
        assert model.num_classes == 6
        assert model.hidden_size == 128
    
    def test_forward_shape(self, device):
        """Test forward pass output shape."""
        model = ResNet1D_GRU(
            num_channels=19,
            num_classes=6
        ).to(device)
        
        # Batch of 4, 19 channels, 10000 time points
        x = torch.randn(4, 19, 10000).to(device)
        
        output = model(x)
        
        # Check output shape
        assert output.shape == (4, 6)
        
        # Check output is valid probabilities
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
        assert torch.allclose(output.sum(dim=1), torch.ones(4).to(device), atol=1e-6)
    
    def test_resnet_block(self):
        """Test ResNet1D block."""
        model = ResNet1D_GRU(num_channels=19, num_classes=6)
        
        # Test first ResNet block
        block = model.resnet_blocks[0]
        x = torch.randn(2, 64, 5000)  # After initial conv
        
        output = block(x)
        
        # Check residual connection
        assert output.shape == x.shape
    
    def test_attention_mechanism(self, device):
        """Test attention mechanism."""
        model = ResNet1D_GRU(
            num_channels=19,
            num_classes=6,
            use_attention=True
        ).to(device)
        
        x = torch.randn(2, 19, 10000).to(device)
        
        # Get intermediate outputs by hooking into forward
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'attention') and module.attention is not None:
                attention_weights.append(output[1])  # Attention weights
        
        # Register hook
        for module in model.modules():
            if hasattr(module, 'gru'):
                module.register_forward_hook(hook_fn)
        
        output = model(x)
        
        # Check that attention was applied
        assert model.use_attention
        assert output.shape == (2, 6)
    
    def test_gradient_flow(self, device):
        """Test gradient flow through model."""
        model = ResNet1D_GRU(
            num_channels=19,
            num_classes=6
        ).to(device)
        
        x = torch.randn(2, 19, 10000, requires_grad=True).to(device)
        target = torch.randint(0, 6, (2,)).to(device)
        
        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert all(p.grad is not None for p in model.parameters())


class TestEfficientNetSpectrogram:
    """Test EfficientNet Spectrogram model."""
    
    def test_init(self):
        """Test model initialization."""
        model = EfficientNetSpectrogram(
            num_classes=6,
            model_name='efficientnet-b0',
            pretrained=False  # For testing
        )
        
        assert model.num_classes == 6
        assert model.model_name == 'efficientnet-b0'
    
    def test_forward_shape(self, device):
        """Test forward pass output shape."""
        model = EfficientNetSpectrogram(
            num_classes=6,
            pretrained=False
        ).to(device)
        
        # Batch of 4, 3 channels (RGB), 224x224 spectrograms
        x = torch.randn(4, 3, 224, 224).to(device)
        
        output = model(x)
        
        # Check output shape
        assert output.shape == (4, 6)
        
        # Check output is valid probabilities
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_multi_task_head(self, device):
        """Test multi-task learning head."""
        model = EfficientNetSpectrogram(
            num_classes=6,
            use_multi_task=True,
            pretrained=False
        ).to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        
        # Should have multi-task heads
        assert hasattr(model, 'seizure_head')
        assert hasattr(model, 'pattern_head')
        
        output = model(x)
        
        # With multi-task, output should be dict
        if isinstance(output, dict):
            assert 'main' in output
            assert 'seizure' in output
            assert 'pattern' in output
            assert output['main'].shape == (2, 6)
        else:
            # Single output
            assert output.shape == (2, 6)
    
    def test_attention_module(self, device):
        """Test channel attention module."""
        model = EfficientNetSpectrogram(
            num_classes=6,
            use_attention=True,
            pretrained=False
        ).to(device)
        
        x = torch.randn(2, 3, 224, 224).to(device)
        output = model(x)
        
        # Check attention was applied
        assert hasattr(model, 'channel_attention')
        assert output.shape == (2, 6)
    
    def test_different_input_sizes(self, device):
        """Test model with different input sizes."""
        model = EfficientNetSpectrogram(
            num_classes=6,
            pretrained=False
        ).to(device)
        
        # Test different input sizes
        sizes = [(224, 224), (256, 256), (384, 384)]
        
        for h, w in sizes:
            x = torch.randn(2, 3, h, w).to(device)
            output = model(x)
            assert output.shape == (2, 6)


class TestHMSEnsembleModel:
    """Test HMS Ensemble model."""
    
    @pytest.fixture
    def mock_base_models(self):
        """Create mock base models."""
        # Mock ResNet model
        resnet_mock = Mock()
        resnet_mock.return_value = torch.randn(2, 6)
        
        # Mock EfficientNet model
        efficientnet_mock = Mock()
        efficientnet_mock.return_value = torch.randn(2, 6)
        
        return {
            'resnet1d_gru': resnet_mock,
            'efficientnet': efficientnet_mock
        }
    
    def test_init(self, mock_base_models):
        """Test ensemble initialization."""
        config = {
            'fusion_method': 'attention',
            'meta_learner_hidden': [128, 64],
            'dropout': 0.3,
            'temperature_scaling': True
        }
        
        model = HMSEnsembleModel(
            base_models=mock_base_models,
            num_classes=6,
            config=config
        )
        
        assert model.num_classes == 6
        assert model.fusion_method == 'attention'
        assert len(model.base_models) == 2
    
    def test_forward_average_fusion(self, device, mock_base_models):
        """Test forward pass with average fusion."""
        config = {'fusion_method': 'average'}
        
        model = HMSEnsembleModel(
            base_models=mock_base_models,
            num_classes=6,
            config=config
        ).to(device)
        
        # Mock inputs
        eeg_input = torch.randn(2, 19, 10000).to(device)
        spec_input = torch.randn(2, 3, 224, 224).to(device)
        
        output = model(eeg_input, spec_input)
        
        # Check output shape
        assert output.shape == (2, 6)
        
        # Check base models were called
        mock_base_models['resnet1d_gru'].assert_called_once()
        mock_base_models['efficientnet'].assert_called_once()
    
    def test_attention_fusion(self, device):
        """Test attention-based fusion."""
        # Create simple base models for testing
        class SimpleModel(nn.Module):
            def __init__(self, output_size):
                super().__init__()
                self.fc = nn.Linear(10, output_size)
            
            def forward(self, x):
                return self.fc(x.mean(dim=-1).mean(dim=-1))
        
        base_models = {
            'model1': SimpleModel(6).to(device),
            'model2': SimpleModel(6).to(device)
        }
        
        config = {'fusion_method': 'attention'}
        
        model = HMSEnsembleModel(
            base_models=base_models,
            num_classes=6,
            config=config
        ).to(device)
        
        # Create inputs
        input1 = torch.randn(2, 10, 100).to(device)
        input2 = torch.randn(2, 10, 100).to(device)
        
        output = model(input1, input2)
        
        # Check output
        assert output.shape == (2, 6)
        
        # Check attention weights sum to 1
        # Note: Would need to expose attention weights for full test
    
    def test_meta_learner(self):
        """Test meta learner component."""
        meta_learner = MetaLearner(
            input_size=12,  # 2 models * 6 classes
            hidden_sizes=[64, 32],
            output_size=6,
            dropout=0.3
        )
        
        # Test forward pass
        x = torch.randn(4, 12)
        output = meta_learner(x)
        
        assert output.shape == (4, 6)
        
        # Check architecture
        assert len(meta_learner.layers) == 6  # 3 linear + 2 dropout + 1 relu
    
    def test_temperature_scaling(self, device, mock_base_models):
        """Test temperature scaling calibration."""
        config = {
            'fusion_method': 'average',
            'temperature_scaling': True
        }
        
        model = HMSEnsembleModel(
            base_models=mock_base_models,
            num_classes=6,
            config=config
        ).to(device)
        
        # Check temperature parameter exists
        assert hasattr(model, 'temperature')
        
        # Test forward pass
        eeg_input = torch.randn(2, 19, 10000).to(device)
        spec_input = torch.randn(2, 3, 224, 224).to(device)
        
        output = model(eeg_input, spec_input)
        assert output.shape == (2, 6)


class TestModelIntegration:
    """Integration tests for model components."""
    
    def test_ensemble_with_real_models(self, device):
        """Test ensemble with actual base models."""
        # Create base models
        resnet = ResNet1D_GRU(
            num_channels=19,
            num_classes=6
        ).to(device)
        
        efficientnet = EfficientNetSpectrogram(
            num_classes=6,
            pretrained=False
        ).to(device)
        
        base_models = {
            'resnet1d_gru': resnet,
            'efficientnet': efficientnet
        }
        
        # Create ensemble
        config = {
            'fusion_method': 'stacking',
            'meta_learner_hidden': [64],
            'dropout': 0.3
        }
        
        ensemble = HMSEnsembleModel(
            base_models=base_models,
            num_classes=6,
            config=config
        ).to(device)
        
        # Test forward pass
        eeg_input = torch.randn(2, 19, 10000).to(device)
        spec_input = torch.randn(2, 3, 224, 224).to(device)
        
        output = ensemble(eeg_input, spec_input)
        
        # Validate output
        assert output.shape == (2, 6)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
        assert torch.allclose(output.sum(dim=1), torch.ones(2).to(device), atol=1e-6)
    
    def test_model_export(self, device):
        """Test model export capabilities."""
        model = ResNet1D_GRU(
            num_channels=19,
            num_classes=6
        ).to(device)
        
        # Test TorchScript export
        x = torch.randn(1, 19, 10000).to(device)
        traced = torch.jit.trace(model, x)
        
        # Test traced model
        output_original = model(x)
        output_traced = traced(x)
        
        assert torch.allclose(output_original, output_traced, atol=1e-6) 