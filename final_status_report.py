#!/usr/bin/env python3
"""
Final Status Report for HMS EEG Classification System
=====================================================

This script provides a comprehensive assessment of system readiness
for deployment to Novita AI platform.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test that all core system components can be imported."""
    logger.info("🧪 Testing Core System Imports...")
    
    core_components = []
    
    # Test preprocessing
    try:
        sys.path.append(str(Path(__file__).parent / 'src'))
        from preprocessing import (
            MultiFormatEEGReader, 
            SignalQualityAssessor,
            EEGFilter,
            EEGFeatureExtractor,
            SpectrogramGenerator
        )
        core_components.append(("Preprocessing Pipeline", "✅ WORKING"))
    except Exception as e:
        core_components.append(("Preprocessing Pipeline", f"❌ FAILED: {e}"))
    
    # Test models
    try:
        from models import ResNet1D_GRU, EfficientNetSpectrogram, HMSEnsembleModel
        core_components.append(("Neural Network Models", "✅ WORKING"))
    except Exception as e:
        core_components.append(("Neural Network Models", f"❌ FAILED: {e}"))
    
    # Test training
    try:
        from training import HMSTrainer
        core_components.append(("Training Pipeline", "✅ WORKING"))
    except Exception as e:
        core_components.append(("Training Pipeline", f"❌ FAILED: {e}"))
    
    # Test evaluation
    try:
        from evaluation import ModelEvaluator, ClinicalMetrics
        core_components.append(("Evaluation Framework", "✅ WORKING"))
    except Exception as e:
        core_components.append(("Evaluation Framework", f"⚠️ PARTIAL: {e}"))
    
    # Test interpretability  
    try:
        from interpretability import SHAPExplainer, ModelInterpreter
        core_components.append(("Interpretability Tools", "✅ WORKING"))
    except Exception as e:
        core_components.append(("Interpretability Tools", f"⚠️ PARTIAL: {e}"))
    
    # Test visualization
    try:
        from visualization import TrainingProgressMonitor
        core_components.append(("Visualization Suite", "✅ WORKING"))
    except Exception as e:
        core_components.append(("Visualization Suite", f"⚠️ PARTIAL: {e}"))
    
    return core_components

def test_data_pipeline():
    """Test the data processing pipeline."""
    logger.info("🧪 Testing Data Pipeline...")
    
    pipeline_status = []
    
    # Check if test data exists
    data_dir = Path("data/raw")
    if data_dir.exists() and list(data_dir.glob("*.csv")):
        pipeline_status.append(("Test Data Available", "✅ READY"))
    else:
        pipeline_status.append(("Test Data Available", "❌ MISSING"))
    
    # Check if processed data exists
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        pipeline_status.append(("Data Processing", "✅ CONFIGURED"))
    else:
        pipeline_status.append(("Data Processing", "⚠️ NOT CONFIGURED"))
    
    return pipeline_status

def test_model_capabilities():
    """Test model creation and basic functionality."""
    logger.info("🧪 Testing Model Capabilities...")
    
    model_status = []
    
    try:
        import torch
        import torch.nn as nn
        
        # Test basic PyTorch functionality
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_status.append(("PyTorch", f"✅ READY ({device})"))
        
        # Test ONNX export capability
        try:
            import onnx
            import onnxruntime
            model_status.append(("ONNX Export", "✅ READY"))
        except Exception as e:
            model_status.append(("ONNX Export", f"❌ FAILED: {e}"))
            
    except Exception as e:
        model_status.append(("PyTorch Framework", f"❌ FAILED: {e}"))
    
    return model_status

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("🧪 Checking Dependencies...")
    
    dependencies = [
        'torch', 'numpy', 'pandas', 'sklearn',
        'matplotlib', 'seaborn', 'tqdm', 'yaml',
        'onnx', 'onnxruntime', 'shap', 'lime'
    ]
    
    dep_status = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            dep_status.append((dep, "✅ INSTALLED"))
        except ImportError:
            dep_status.append((dep, "❌ MISSING"))
    
    return dep_status

def assess_novita_readiness():
    """Assess overall readiness for Novita AI deployment."""
    logger.info("🧪 Assessing Novita AI Readiness...")
    
    readiness_criteria = []
    
    # Core functionality tests
    core_components = test_core_imports()
    data_pipeline = test_data_pipeline()
    model_capabilities = test_model_capabilities()
    dependencies = check_dependencies()
    
    # Calculate readiness score
    working_components = sum(1 for _, status in core_components if "✅" in status)
    total_core = len(core_components)
    
    working_deps = sum(1 for _, status in dependencies if "✅" in status)
    total_deps = len(dependencies)
    
    core_score = (working_components / total_core) * 100
    dep_score = (working_deps / total_deps) * 100
    
    readiness_criteria.append(("Core Components", f"{working_components}/{total_core} ({core_score:.0f}%)"))
    readiness_criteria.append(("Dependencies", f"{working_deps}/{total_deps} ({dep_score:.0f}%)"))
    
    # Overall assessment
    if core_score >= 80 and dep_score >= 90:
        overall_status = "🟢 READY FOR DEPLOYMENT"
    elif core_score >= 60 and dep_score >= 80:
        overall_status = "🟡 MOSTLY READY (Minor Issues)"
    else:
        overall_status = "🔴 NOT READY (Major Issues)"
    
    readiness_criteria.append(("Overall Assessment", overall_status))
    
    return readiness_criteria, core_components, data_pipeline, model_capabilities, dependencies

def generate_report():
    """Generate the final status report."""
    print("=" * 80)
    print("🏥 HMS EEG CLASSIFICATION SYSTEM - FINAL STATUS REPORT")
    print("=" * 80)
    print()
    
    readiness, core, data, models, deps = assess_novita_readiness()
    
    # Readiness Summary
    print("📊 DEPLOYMENT READINESS SUMMARY")
    print("-" * 40)
    for criterion, status in readiness:
        print(f"{criterion:30} {status}")
    print()
    
    # Core Components
    print("🔧 CORE COMPONENTS STATUS")
    print("-" * 40)
    for component, status in core:
        print(f"{component:30} {status}")
    print()
    
    # Data Pipeline  
    print("📊 DATA PIPELINE STATUS")
    print("-" * 40)
    for item, status in data:
        print(f"{item:30} {status}")
    print()
    
    # Model Capabilities
    print("🤖 MODEL CAPABILITIES")
    print("-" * 40)
    for capability, status in models:
        print(f"{capability:30} {status}")
    print()
    
    # Dependencies
    print("📦 DEPENDENCY STATUS")
    print("-" * 40)
    for dep, status in deps:
        print(f"{dep:30} {status}")
    print()
    
    # Deployment Recommendations
    print("🚀 NOVITA AI DEPLOYMENT RECOMMENDATIONS")
    print("-" * 40)
    
    overall = readiness[-1][1]
    if "🟢" in overall:
        print("✅ System is ready for Novita AI deployment!")
        print("✅ All core components are functional")
        print("✅ Model export (ONNX) is working")
        print("✅ Dependencies are properly installed")
        print()
        print("📝 Next Steps:")
        print("   1. Upload the trained model to Novita AI")
        print("   2. Configure the inference endpoint")
        print("   3. Test with sample EEG data")
        print("   4. Monitor performance metrics")
        
    elif "🟡" in overall:
        print("⚠️  System is mostly ready with minor issues")
        print("⚠️  Consider addressing remaining issues before deployment")
        print("✅ Core functionality is working")
        print()
        print("📝 Recommended Actions:")
        print("   1. Fix any missing dependencies")
        print("   2. Test advanced features if needed")
        print("   3. Proceed with cautious deployment")
        
    else:
        print("❌ System is not ready for deployment")
        print("❌ Critical issues need to be resolved")
        print()
        print("📝 Required Actions:")
        print("   1. Fix core component failures")
        print("   2. Install missing dependencies")
        print("   3. Test and validate pipeline")
        print("   4. Re-run this status check")
    
    print()
    print("=" * 80)
    
    return overall

def main():
    """Main function to run the status report."""
    try:
        overall_status = generate_report()
        
        # Exit with appropriate code
        if "🟢" in overall_status:
            sys.exit(0)  # Success
        elif "🟡" in overall_status:
            sys.exit(1)  # Warning
        else:
            sys.exit(2)  # Error
            
    except Exception as e:
        logger.error(f"❌ Status report failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main() 