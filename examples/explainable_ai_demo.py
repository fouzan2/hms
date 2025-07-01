#!/usr/bin/env python3
"""
Explainable AI with Counterfactual Reasoning Demonstration for HMS Brain Activity Classification
Comprehensive demo showing counterfactual explanations, SHAP integration, gradient methods,
attention visualization, and clinical interpretation tools.
"""

import sys
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def generate_demo_eeg_data(n_samples=20, n_channels=19, seq_length=2000):
    """Generate synthetic EEG data for demonstration."""
    print(f"Generating {n_samples} synthetic EEG samples...")
    
    # Generate realistic EEG-like signals
    t = np.linspace(0, seq_length/200, seq_length)  # 200 Hz sampling rate
    eeg_data = []
    labels = []
    class_names = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other']
    
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
            label = i % 6
            if label == 0:  # Seizure-like pattern
                seizure_pattern = 3.0 * np.sin(2 * np.pi * 3 * t) * np.exp(-t/8)
                channels[ch] = alpha + beta + theta + gamma + noise + seizure_pattern
            elif label == 1:  # LPD pattern
                lpd_pattern = 1.5 * np.sin(2 * np.pi * 1.5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.2 * t))
                channels[ch] = alpha + beta + theta + gamma + noise + lpd_pattern
            elif label == 2:  # GPD pattern
                gpd_pattern = 2.0 * np.sin(2 * np.pi * 2 * t) * np.abs(np.sin(2 * np.pi * 0.3 * t))
                channels[ch] = alpha + beta + theta + gamma + noise + gpd_pattern
            else:
                channels[ch] = alpha + beta + theta + gamma + noise
            
        eeg_data.append(channels)
        labels.append(label)
        
    return eeg_data, labels, class_names

def create_demo_model():
    """Create a simple demonstration model."""
    class DemoEEGModel(torch.nn.Module):
        def __init__(self, n_channels=19, n_classes=6):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=5, padding=2)
            self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=3, padding=1)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.fc1 = torch.nn.Linear(256, 128)
            self.fc2 = torch.nn.Linear(128, n_classes)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(0.3)
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x).squeeze(-1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return DemoEEGModel()

def demo_explainable_ai_framework():
    """Demonstrate the complete Explainable AI framework."""
    print("\n" + "="*80)
    print("🔍 EXPLAINABLE AI WITH COUNTERFACTUAL REASONING DEMO")
    print("="*80)
    
    try:
        from interpretability import (
            create_explainable_ai,
            ExplanationConfig,
            CounterfactualGenerator,
            SHAPExplainer,
            ClinicalInterpreter,
            create_gradient_explainer
        )
        
        # Create demo model and data
        print("\n📊 Setting up demonstration environment...")
        model = create_demo_model()
        model.eval()
        
        # Generate demo data
        eeg_data, labels, class_names = generate_demo_eeg_data(n_samples=15)
        
        # Split data
        background_data = torch.tensor(eeg_data[:10], dtype=torch.float32)
        test_samples = torch.tensor(eeg_data[10:], dtype=torch.float32)
        test_labels = labels[10:]
        
        print(f"  Generated {len(eeg_data)} EEG samples")
        print(f"  Background data: {background_data.shape}")
        print(f"  Test samples: {test_samples.shape}")
        print(f"  Classes: {class_names}")
        
        # Create explainer with custom configuration
        config = ExplanationConfig(
            cf_max_iterations=100,  # Reduced for demo speed
            cf_learning_rate=0.02,
            shap_method='gradient',  # Faster than deep SHAP
            shap_n_samples=20,
            frequency_bands={
                'delta': (0.5, 4.0),
                'theta': (4.0, 8.0), 
                'alpha': (8.0, 13.0),
                'beta': (13.0, 30.0),
                'gamma': (30.0, 50.0)
            }
        )
        
        explainer = create_explainable_ai(model, config, device='cpu')
        explainer.initialize_background_data(background_data)
        
        print("✅ Explainable AI framework initialized")
        
        return explainer, test_samples, test_labels, class_names, config
        
    except Exception as e:
        print(f"❌ Framework setup failed: {e}")
        return None, None, None, None, None

def demo_comprehensive_explanation():
    """Demonstrate comprehensive explanation generation."""
    print("\n" + "="*60)
    print("🎯 COMPREHENSIVE EXPLANATION GENERATION")
    print("="*60)
    
    explainer, test_samples, test_labels, class_names, config = demo_explainable_ai_framework()
    
    if explainer is None:
        return
    
    # Select test sample
    sample_idx = 0
    test_sample = test_samples[sample_idx:sample_idx+1]
    true_label = test_labels[sample_idx]
    
    print(f"\n🔬 Analyzing sample {sample_idx + 1}")
    print(f"  True class: {class_names[true_label]}")
    print(f"  Sample shape: {test_sample.shape}")
    
    # Generate comprehensive explanation
    print("\n⚙️ Generating comprehensive explanation...")
    explanation = explainer.explain_prediction(
        test_sample,
        explanation_types=['prediction', 'counterfactual', 'shap', 'clinical'],
        target_class=None  # Let it choose automatically
    )
    
    # Display results
    print("\n📊 PREDICTION RESULTS:")
    pred_info = explanation['prediction']
    print(f"  Predicted class: {pred_info['predicted_class_idx']} ({class_names[pred_info['predicted_class_idx']]})")
    print(f"  Confidence: {pred_info['confidence']:.3f}")
    print(f"  Correct: {'✅' if pred_info['predicted_class_idx'] == true_label else '❌'}")
    
    # Top predictions
    if 'top_k_predictions' in pred_info:
        values, indices = pred_info['top_k_predictions']
        print(f"  Top predictions:")
        for i, (val, idx) in enumerate(zip(values, indices)):
            print(f"    {i+1}. {class_names[idx]}: {val:.3f}")
    
    # SHAP Analysis
    if 'shap' in explanation and 'error' not in explanation['shap']:
        print(f"\n🎯 SHAP FEATURE IMPORTANCE:")
        shap_analysis = explanation['shap']['analysis']
        print(f"  Global importance: {shap_analysis['global_importance']:.6f}")
        
        # Top channels
        channel_importance = shap_analysis['channel_importance']
        top_channels = np.argsort(channel_importance)[-3:][::-1]
        print(f"  Most important channels:")
        for ch in top_channels:
            print(f"    Channel {ch}: {channel_importance[ch]:.6f}")
    
    # Counterfactual Analysis
    if 'counterfactual' in explanation and 'error' not in explanation['counterfactual']:
        print(f"\n🔄 COUNTERFACTUAL ANALYSIS:")
        cf = explanation['counterfactual']
        print(f"  Target class: {class_names[cf['target_class']]}")
        print(f"  Success: {'✅' if cf['success'] else '❌'}")
        
        if cf['success']:
            print(f"  Total change required: {cf['total_change']:.6f}")
            print(f"  Relative change: {cf['relative_change']:.3f}")
            print(f"  New confidence: {cf['confidence']:.3f}")
            
            if cf['relative_change'] < 0.1:
                print(f"  💡 Insight: Small changes could alter the diagnosis")
            elif cf['relative_change'] > 0.5:
                print(f"  💡 Insight: Large changes needed - robust prediction")
    
    # Clinical Interpretation
    if 'clinical' in explanation and 'error' not in explanation['clinical']:
        print(f"\n🏥 CLINICAL INTERPRETATION:")
        clinical = explanation['clinical']
        print(f"  Summary: {clinical['summary']}")
        
        if clinical['recommendations']:
            print(f"  Recommendations:")
            for rec in clinical['recommendations']:
                print(f"    - {rec}")
        
        if clinical['risk_factors']:
            print(f"  Risk factors:")
            for factor in clinical['risk_factors']:
                print(f"    - {factor}")
        
        # Confidence analysis
        conf_analysis = clinical['confidence_analysis']
        print(f"  Confidence category: {conf_analysis['confidence_category']}")
    
    return explanation, test_sample

def demo_counterfactual_visualization():
    """Demonstrate counterfactual visualization."""
    print("\n" + "="*60)
    print("🔄 COUNTERFACTUAL VISUALIZATION")
    print("="*60)
    
    explanation, test_sample = demo_comprehensive_explanation()
    
    if explanation is None or 'counterfactual' not in explanation:
        print("❌ No counterfactual explanation available")
        return
    
    cf = explanation['counterfactual']
    if not cf['success']:
        print("❌ Counterfactual generation was not successful")
        return
    
    print(f"\n📊 Visualizing counterfactual changes...")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Counterfactual Analysis Visualization', fontsize=16)
    
    # Original EEG
    original = cf['original'].numpy().squeeze()
    im1 = axes[0, 0].imshow(original, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Original EEG')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Channels')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Counterfactual EEG
    counterfactual = cf['counterfactual'].numpy().squeeze()
    im2 = axes[0, 1].imshow(counterfactual, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Counterfactual EEG')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Channels')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Changes (difference)
    changes = cf['changes'].numpy().squeeze()
    max_change = np.max(np.abs(changes))
    im3 = axes[1, 0].imshow(changes, aspect='auto', cmap='RdBu_r', 
                           vmin=-max_change, vmax=max_change)
    axes[1, 0].set_title('Changes Required')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Channels')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Optimization progress
    if 'optimization_losses' in cf:
        axes[1, 1].plot(cf['optimization_losses'])
        axes[1, 1].set_title('Optimization Progress')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('counterfactual_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Counterfactual visualization saved as 'counterfactual_analysis.png'")

def demo_shap_analysis():
    """Demonstrate SHAP feature importance analysis."""
    print("\n" + "="*60)
    print("🎯 SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    explanation, test_sample = demo_comprehensive_explanation()
    
    if explanation is None or 'shap' not in explanation:
        print("❌ No SHAP explanation available")
        return
    
    shap_data = explanation['shap']
    if 'error' in shap_data:
        print(f"❌ SHAP analysis failed: {shap_data['error']}")
        return
    
    print(f"\n📊 Visualizing SHAP importance...")
    
    # Extract SHAP values and analysis
    shap_values = shap_data['shap_values']
    analysis = shap_data['analysis']
    
    # Create SHAP visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('SHAP Feature Importance Analysis', fontsize=16)
    
    # SHAP values heatmap
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[0])  # Take first class
    else:
        shap_array = shap_values
    
    if len(shap_array.shape) > 2:
        shap_array = shap_array[0]  # Take first sample
    
    max_shap = np.max(np.abs(shap_array))
    im1 = axes[0, 0].imshow(shap_array, aspect='auto', cmap='RdBu_r',
                           vmin=-max_shap, vmax=max_shap)
    axes[0, 0].set_title('SHAP Values')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Channels')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Channel importance
    channel_importance = analysis['channel_importance']
    axes[0, 1].barh(range(len(channel_importance)), channel_importance)
    axes[0, 1].set_title('Channel Importance')
    axes[0, 1].set_xlabel('Importance')
    axes[0, 1].set_ylabel('Channel')
    axes[0, 1].invert_yaxis()
    
    # Temporal importance
    temporal_importance = analysis['temporal_importance']
    time_points = np.arange(len(temporal_importance))
    axes[1, 0].plot(time_points, temporal_importance)
    axes[1, 0].set_title('Temporal Importance')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Frequency band importance
    freq_importance = analysis['frequency_importance']
    bands = list(freq_importance.keys())
    values = list(freq_importance.values())
    
    bars = axes[1, 1].bar(bands, values)
    axes[1, 1].set_title('Frequency Band Importance')
    axes[1, 1].set_xlabel('Frequency Band')
    axes[1, 1].set_ylabel('Importance')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('shap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ SHAP analysis visualization saved as 'shap_analysis.png'")

def demo_gradient_explanations():
    """Demonstrate gradient-based explanation methods."""
    print("\n" + "="*60)
    print("🌊 GRADIENT-BASED EXPLANATIONS")
    print("="*60)
    
    try:
        from interpretability import create_gradient_explainer
        
        # Create demo setup
        model = create_demo_model()
        model.eval()
        
        eeg_data, _, _ = generate_demo_eeg_data(n_samples=1, seq_length=1000)
        test_sample = torch.tensor(eeg_data[0], dtype=torch.float32).unsqueeze(0)
        
        # Create gradient explainer
        target_layers = ['conv1', 'conv2', 'conv3']
        grad_explainer = create_gradient_explainer(model, target_layers, device='cpu')
        
        print(f"📊 Generating gradient explanations...")
        
        # Generate multiple gradient explanations
        methods = ['integrated_gradients', 'guided_backprop', 'smooth_grad']
        explanations = grad_explainer.explain(
            test_sample,
            methods=methods,
            ig_params={'steps': 20},  # Reduced for demo
            sg_params={'n_samples': 20, 'noise_level': 0.1}
        )
        
        # Compare methods
        comparison = grad_explainer.compare_methods(explanations)
        
        print(f"\n📈 Method Comparison:")
        for method, metrics in comparison.items():
            if method in explanations and 'error' not in explanations[method]:
                print(f"  {method}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
        
        # Visualize explanations
        print(f"\n📊 Creating gradient explanation visualizations...")
        
        fig = grad_explainer.visualize_explanations(
            explanations,
            test_sample,
            save_path='gradient_explanations.png'
        )
        
        print(f"✅ Gradient explanations saved as 'gradient_explanations.png'")
        
    except Exception as e:
        print(f"❌ Gradient explanations demo failed: {e}")

def demo_clinical_interpretation():
    """Demonstrate clinical interpretation capabilities."""
    print("\n" + "="*60)
    print("🏥 CLINICAL INTERPRETATION")
    print("="*60)
    
    explanation, test_sample = demo_comprehensive_explanation()
    
    if explanation is None or 'clinical' not in explanation:
        print("❌ No clinical interpretation available")
        return
    
    clinical = explanation['clinical']
    if 'error' in clinical:
        print(f"❌ Clinical interpretation failed: {clinical['error']}")
        return
    
    print(f"\n📊 Detailed Clinical Analysis:")
    
    # Clinical features
    clinical_features = clinical['clinical_features']
    
    # Power spectral density
    if 'power_spectral_density' in clinical_features:
        print(f"\n🌊 Power Spectral Density:")
        psd_features = clinical_features['power_spectral_density']
        for band, features in psd_features.items():
            mean_power = features['mean_power']
            std_power = features['std_power']
            print(f"  {band.capitalize()}: {mean_power:.4f} ± {std_power:.4f}")
    
    # Spectral entropy
    if 'spectral_entropy' in clinical_features:
        entropy_features = clinical_features['spectral_entropy']
        print(f"\n📊 Spectral Entropy:")
        print(f"  Mean: {entropy_features['mean_entropy']:.4f}")
        print(f"  Std: {entropy_features['std_entropy']:.4f}")
    
    # Hjorth parameters
    if 'hjorth_parameters' in clinical_features:
        print(f"\n📈 Hjorth Parameters (per channel):")
        hjorth_params = clinical_features['hjorth_parameters']
        for i, params in enumerate(hjorth_params[:3]):  # Show first 3 channels
            print(f"  Channel {i}:")
            print(f"    Activity: {params['activity']:.4f}")
            print(f"    Mobility: {params['mobility']:.4f}")
            print(f"    Complexity: {params['complexity']:.4f}")
    
    # Connectivity
    if 'connectivity' in clinical_features:
        connectivity = clinical_features['connectivity']
        print(f"\n🔗 Connectivity Analysis:")
        print(f"  Average connectivity: {connectivity['average_connectivity']:.4f}")
        print(f"  Max connectivity: {connectivity['max_connectivity']:.4f}")
        print(f"  Connectivity variance: {connectivity['connectivity_variance']:.4f}")
    
    # Confidence analysis
    conf_analysis = clinical['confidence_analysis']
    print(f"\n🎯 Confidence Analysis:")
    print(f"  Confidence level: {conf_analysis['confidence_level']:.3f}")
    print(f"  Category: {conf_analysis['confidence_category']}")
    print(f"  Reliability indicators:")
    for indicator in conf_analysis['reliability_indicators']:
        print(f"    - {indicator}")

def demo_comprehensive_report():
    """Demonstrate comprehensive explanation report generation."""
    print("\n" + "="*60)
    print("📄 COMPREHENSIVE EXPLANATION REPORT")
    print("="*60)
    
    explanation, test_sample = demo_comprehensive_explanation()
    
    if explanation is None:
        print("❌ No explanation available for report")
        return
    
    # Generate comprehensive report
    from interpretability import create_explainable_ai
    
    model = create_demo_model()
    explainer = create_explainable_ai(model, device='cpu')
    
    print(f"\n📊 Generating comprehensive report...")
    
    report = explainer.generate_explanation_report(
        explanation,
        save_path='explanation_report.txt'
    )
    
    # Display excerpt
    print(f"\n📋 Report Excerpt:")
    print("-" * 50)
    print(report[:800] + "..." if len(report) > 800 else report)
    print("-" * 50)
    
    print(f"\n✅ Full report saved as 'explanation_report.txt'")
    print(f"📊 Report length: {len(report)} characters")

def main():
    """Run all explainable AI demonstrations."""
    print("🔍 Explainable AI with Counterfactual Reasoning Demonstration")
    print("HMS Brain Activity Classification Project")
    print("=" * 80)
    
    # Run demonstrations
    try:
        # 1. Framework overview
        demo_explainable_ai_framework()
        
        # 2. Comprehensive explanation
        demo_comprehensive_explanation()
        
        # 3. Counterfactual visualization
        demo_counterfactual_visualization()
        
        # 4. SHAP analysis
        demo_shap_analysis()
        
        # 5. Gradient explanations
        demo_gradient_explanations()
        
        # 6. Clinical interpretation
        demo_clinical_interpretation()
        
        # 7. Comprehensive report
        demo_comprehensive_report()
        
        print("\n" + "="*80)
        print("🎉 EXPLAINABLE AI DEMONSTRATION COMPLETE!")
        print("="*80)
        print("Key Outputs Generated:")
        print("  - counterfactual_analysis.png")
        print("  - shap_analysis.png") 
        print("  - gradient_explanations.png")
        print("  - explanation_report.txt")
        print("")
        print("Key Features Demonstrated:")
        print("  ✅ Counterfactual 'what-if' scenario generation")
        print("  ✅ SHAP-based feature importance analysis")
        print("  ✅ Multiple gradient explanation methods")
        print("  ✅ Clinical interpretation and recommendations")
        print("  ✅ Interactive visualizations and reports")
        print("  ✅ Integration with EEG preprocessing and models")
        print("")
        print("🔍 The explainable AI framework provides comprehensive")
        print("   interpretability for all HMS model predictions!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 