#!/usr/bin/env python3
"""
EEG Foundation Model Demonstration for HMS Brain Activity Classification
Shows how to use the foundation model for various tasks including:
- Model creation and configuration
- Pre-training simulation
- Fine-tuning for classification
- Feature extraction
- Integration with preprocessing
"""

import sys
import numpy as np
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def generate_demo_eeg_data(n_samples=50, n_channels=19, seq_length=2000):
    """Generate synthetic EEG data for demonstration."""
    print(f"Generating {n_samples} synthetic EEG samples...")
    
    # Generate realistic EEG-like signals
    t = np.linspace(0, seq_length/200, seq_length)  # 200 Hz sampling rate
    eeg_data = []
    labels = []
    
    for i in range(n_samples):
        # Generate multi-channel EEG
        channels = np.zeros((n_channels, seq_length))
        
        for ch in range(n_channels):
            # Mix of different frequency bands with random phases
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            beta = 0.3 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            theta = 0.4 * np.sin(2 * np.pi * 6 * t + np.random.rand() * 2 * np.pi)
            gamma = 0.2 * np.sin(2 * np.pi * 40 * t + np.random.rand() * 2 * np.pi)
            
            # Add some noise
            noise = 0.1 * np.random.randn(seq_length)
            
            # Different patterns for different classes
            if i % 6 == 0:  # Seizure-like pattern
                seizure_pattern = 2.0 * np.sin(2 * np.pi * 3 * t) * np.exp(-t/5)
                channels[ch] = alpha + beta + theta + gamma + noise + seizure_pattern
            else:
                channels[ch] = alpha + beta + theta + gamma + noise
            
        eeg_data.append(channels)
        labels.append(i % 6)  # 6 classes: seizure, lpd, gpd, lrda, grda, other
        
    return eeg_data, labels

def demo_model_creation():
    """Demonstrate EEG Foundation Model creation."""
    print("\n" + "="*60)
    print("🧠 EEG Foundation Model Creation Demo")
    print("="*60)
    
    from models import EEGFoundationModel, EEGFoundationConfig, create_model
    
    # Create different model sizes
    configs = {
        'Small': EEGFoundationConfig(
            d_model=128,
            n_heads=4,
            n_layers=6,
            n_channels=19
        ),
        'Medium': EEGFoundationConfig(
            d_model=256,
            n_heads=8,
            n_layers=9,
            n_channels=19
        ),
        'Large': EEGFoundationConfig(
            d_model=512,
            n_heads=16,
            n_layers=12,
            n_channels=19
        )
    }
    
    for size, config in configs.items():
        print(f"\n📊 Creating {size} Model:")
        model = EEGFoundationModel(config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test forward pass
        test_input = torch.randn(2, 19, 2000)  # 2 samples, 19 channels, 10s at 200Hz
        
        with torch.no_grad():
            outputs = model(test_input)
            print(f"  Output shape: {outputs['pooled_output'].shape}")
            print(f"  Contrastive features shape: {outputs['contrastive_features'].shape}")
        
        del model  # Free memory
    
    # Test factory function
    print(f"\n🏭 Testing Factory Function:")
    model = create_model('eeg_foundation', {
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'n_channels': 19
    })
    print(f"  Factory-created model parameters: {sum(p.numel() for p in model.parameters()):,}")

def demo_pretraining_simulation():
    """Demonstrate pre-training simulation."""
    print("\n" + "="*60)
    print("🔄 Pre-training Simulation Demo")
    print("="*60)
    
    from models import EEGFoundationModel, EEGFoundationConfig, EEGFoundationPreTrainer
    
    # Create small model for demo
    config = EEGFoundationConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_channels=19,
        mask_ratio=0.15
    )
    
    model = EEGFoundationModel(config)
    pre_trainer = EEGFoundationPreTrainer(model, device='cpu')
    
    print(f"📝 Model Configuration:")
    print(f"  Mask ratio: {config.mask_ratio}")
    print(f"  Reconstruction weight: {config.reconstruction_weight}")
    print(f"  Contrastive weight: {config.contrastive_weight}")
    
    # Generate demo data
    eeg_data, _ = generate_demo_eeg_data(n_samples=4, seq_length=1000)
    eeg_tensor = torch.tensor(np.array(eeg_data), dtype=torch.float32)
    
    print(f"\n🧪 Pre-training Steps:")
    for step in range(3):
        total_loss, losses = pre_trainer.pretrain_step(eeg_tensor)
        
        print(f"  Step {step + 1}:")
        print(f"    Total loss: {total_loss.item():.4f}")
        for loss_name, loss_value in losses.items():
            if loss_name != 'total_loss':
                print(f"    {loss_name}: {loss_value:.4f}")

def demo_finetuning():
    """Demonstrate fine-tuning for classification."""
    print("\n" + "="*60)
    print("🎯 Fine-tuning Demo")
    print("="*60)
    
    from models import (
        EEGFoundationModel, EEGFoundationConfig, 
        EEGFoundationTrainer, FineTuningConfig,
        EEGDataset
    )
    from torch.utils.data import DataLoader
    
    # Create model
    foundation_config = EEGFoundationConfig(
        d_model=128,
        n_heads=4,
        n_layers=3,
        n_channels=19
    )
    
    model = EEGFoundationModel(foundation_config)
    
    # Add classification head
    num_classes = 6
    class_names = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    model.add_classification_head(num_classes)
    
    print(f"📊 Classification Setup:")
    print(f"  Number of classes: {num_classes}")
    print(f"  Class names: {class_names}")
    
    # Generate demo data
    train_data, train_labels = generate_demo_eeg_data(n_samples=20, seq_length=1000)
    val_data, val_labels = generate_demo_eeg_data(n_samples=8, seq_length=1000)
    
    # Create datasets and loaders
    train_dataset = EEGDataset(train_data, train_labels)
    val_dataset = EEGDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    # Test classification forward pass
    test_input = torch.randn(2, 19, 1000)
    with torch.no_grad():
        logits = model.classify(test_input)
        probabilities = torch.softmax(logits, dim=1)
        
    print(f"\n🔍 Model Output:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Probabilities shape: {probabilities.shape}")
    print(f"  Sample predictions: {torch.argmax(probabilities, dim=1).tolist()}")

def demo_feature_extraction():
    """Demonstrate feature extraction capabilities."""
    print("\n" + "="*60)
    print("🎨 Feature Extraction Demo")
    print("="*60)
    
    from models import EEGFoundationModel, EEGFoundationConfig
    
    # Create model
    config = EEGFoundationConfig(
        d_model=256,
        n_heads=8,
        n_layers=6,
        n_channels=19
    )
    
    model = EEGFoundationModel(config)
    
    # Generate demo data
    eeg_data, labels = generate_demo_eeg_data(n_samples=10, seq_length=2000)
    eeg_tensor = torch.tensor(np.array(eeg_data), dtype=torch.float32)
    
    print(f"📊 Extracting Features:")
    print(f"  Input shape: {eeg_tensor.shape}")
    
    # Extract embeddings
    with torch.no_grad():
        embeddings = model.get_embeddings(eeg_tensor)
        
        # Get detailed outputs
        outputs = model(eeg_tensor, return_hidden_states=True)
        
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Hidden states: {len(outputs.get('hidden_states', []))} layers")
    
    # Demonstrate different use cases for embeddings
    print(f"\n🔬 Embedding Analysis:")
    
    # Compute similarity between samples
    similarities = torch.mm(embeddings, embeddings.t())
    print(f"  Similarity matrix shape: {similarities.shape}")
    
    # Group by class
    unique_labels = sorted(set(labels))
    print(f"  Classes present: {unique_labels}")
    
    for label in unique_labels:
        class_indices = [i for i, l in enumerate(labels) if l == label]
        if len(class_indices) > 1:
            class_embeddings = embeddings[class_indices]
            class_mean = torch.mean(class_embeddings, dim=0)
            class_std = torch.std(class_embeddings, dim=0)
            print(f"  Class {label}: {len(class_indices)} samples, "
                  f"mean norm: {torch.norm(class_mean):.3f}, "
                  f"std norm: {torch.norm(class_std):.3f}")

def demo_integration_with_preprocessing():
    """Demonstrate integration with adaptive preprocessing."""
    print("\n" + "="*60)
    print("🔧 Integration with Adaptive Preprocessing Demo")
    print("="*60)
    
    try:
        from models import EEGFoundationModel, EEGFoundationConfig
        from preprocessing import EEGPreprocessor
        
        # Create foundation model
        config = EEGFoundationConfig(
            d_model=128,
            n_heads=4,
            n_layers=3,
            n_channels=19
        )
        
        model = EEGFoundationModel(config)
        
        # Create preprocessor with adaptive enabled
        preprocessor = EEGPreprocessor(use_adaptive=True)
        
        # Generate test data
        eeg_data, _ = generate_demo_eeg_data(n_samples=1, seq_length=2000)
        channel_names = [f'CH{i+1}' for i in range(19)]
        
        print(f"📊 Processing Pipeline:")
        print(f"  Raw EEG shape: {eeg_data[0].shape}")
        
        # Preprocess data
        processed_data, preprocessing_info = preprocessor.preprocess_eeg(
            eeg_data[0], channel_names
        )
        
        print(f"  Processed shape: {processed_data.shape}")
        print(f"  Preprocessing method: {preprocessing_info.get('method', 'unknown')}")
        
        # Use with foundation model
        test_input = torch.tensor(processed_data, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(test_input)
            
        print(f"  Foundation model output: {outputs['pooled_output'].shape}")
        print(f"✅ Integration successful!")
        
    except Exception as e:
        print(f"⚠️ Integration demo failed: {e}")
        print("This might be due to missing preprocessing components.")

def demo_transfer_learning_pipeline():
    """Demonstrate transfer learning pipeline."""
    print("\n" + "="*60)
    print("📚 Transfer Learning Pipeline Demo")
    print("="*60)
    
    from models import TransferLearningPipeline, FineTuningConfig
    
    # Create pipeline config
    config = FineTuningConfig(
        epochs=2,  # Very short for demo
        batch_size=4,
        learning_rate=1e-3,
        early_stopping=False,
        save_best_model=False,
        save_last_model=False
    )
    
    # Create pipeline (without pre-trained model for demo)
    pipeline = TransferLearningPipeline(config=config)
    
    print(f"📊 Pipeline Configuration:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    
    # Generate demo dataset
    print(f"\n📝 Generating Demo Dataset:")
    train_data, train_labels = generate_demo_eeg_data(n_samples=16, seq_length=800)
    val_data, val_labels = generate_demo_eeg_data(n_samples=8, seq_length=800)
    test_data, test_labels = generate_demo_eeg_data(n_samples=4, seq_length=800)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    # Note: Actually running the pipeline would take too long for a demo
    print(f"\n💡 Note: Full pipeline execution takes significant time.")
    print(f"   For demonstration, we're showing the setup only.")
    print(f"   To run the full pipeline, use:")
    print(f"   results = pipeline.run_pipeline(train_data, train_labels, ...)")

def demo_model_variants():
    """Demonstrate different model variants."""
    print("\n" + "="*60)
    print("🏗️ Model Variants Demo")
    print("="*60)
    
    from models import EEGFoundationConfig
    
    # Define model variants
    variants = {
        'Tiny': {
            'd_model': 64,
            'n_heads': 2,
            'n_layers': 3,
            'n_channels': 19,
            'max_seq_length': 2000
        },
        'Small': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 6,
            'n_channels': 19,
            'max_seq_length': 5000
        },
        'Medium': {
            'd_model': 256,
            'n_heads': 8,
            'n_layers': 9,
            'n_channels': 19,
            'max_seq_length': 8000
        },
        'Large': {
            'd_model': 512,
            'n_heads': 16,
            'n_layers': 12,
            'n_channels': 19,
            'max_seq_length': 10000
        }
    }
    
    print(f"📊 Model Variant Comparison:")
    print(f"{'Variant':<8} {'Params':<10} {'Memory':<8} {'Seq Len':<8}")
    print("-" * 40)
    
    for name, config_dict in variants.items():
        config = EEGFoundationConfig(**config_dict)
        
        # Estimate parameters (simplified calculation)
        d_model = config.d_model
        n_layers = config.n_layers
        n_heads = config.n_heads
        d_ff = config.d_ff
        
        # Rough parameter estimation
        embedding_params = config.n_channels * d_model + config.max_position_embeddings * d_model
        transformer_params = n_layers * (
            3 * d_model * d_model +  # QKV projections
            d_model * d_ff * 2 +     # FFN
            d_model * 4              # Layer norms and biases
        )
        total_params = embedding_params + transformer_params
        
        memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        
        print(f"{name:<8} {total_params/1e6:>7.1f}M {memory_mb:>6.1f}MB {config.max_seq_length:>7}")

def main():
    """Run all demonstrations."""
    print("🎭 EEG Foundation Model Comprehensive Demo")
    print("HMS Brain Activity Classification Project")
    print("=" * 60)
    
    # Run all demos
    demo_model_creation()
    demo_pretraining_simulation()
    demo_finetuning()
    demo_feature_extraction()
    demo_integration_with_preprocessing()
    demo_transfer_learning_pipeline()
    demo_model_variants()
    
    print("\n" + "="*60)
    print("🎉 Demo Complete!")
    print("="*60)
    print("Key Takeaways:")
    print("- ✅ EEG Foundation Model supports multiple architectures")
    print("- ✅ Self-supervised pre-training with masked modeling")
    print("- ✅ Fine-tuning for downstream classification tasks")
    print("- ✅ Feature extraction for various applications")
    print("- ✅ Integration with adaptive preprocessing")
    print("- ✅ Transfer learning pipeline for end-to-end training")
    print("- ✅ Multiple model variants for different use cases")
    print("\nFor more details, see: EEG_FOUNDATION_MODEL_GUIDE.md")

if __name__ == "__main__":
    main() 