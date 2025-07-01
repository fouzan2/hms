#!/bin/bash
set -e

echo "🧪 HMS Pipeline Local Test Suite"
echo "================================"

# Step 1: Setup
echo "📦 Step 1: Environment Setup"
chmod +x quick_setup.sh validation_checklist.sh
# ./quick_setup.sh  # Uncomment if you need to install packages

# Step 2: Create test data
echo "📊 Step 2: Creating Test Data"
python test_data_prep.py

# Step 3: Run diagnostic checks
echo "🔧 Step 3: Diagnostic Checks"
python debug_fixes.py

# Step 4: Test individual components
echo "🧩 Step 4: Component Tests"
echo "🧪 Testing core preprocessing pipeline..."
python test_preprocessing.py

echo "🧪 Testing adaptive preprocessing..."
python test_adaptive_preprocessing.py

echo "🧪 Testing adaptive preprocessing integration..."
python test_adaptive_integration.py

echo "🧪 Testing EEG Foundation Model..."
python test_eeg_foundation_model.py

echo "🧪 Testing explainable AI and interpretability..."
python test_interpretability.py

echo "🧪 Testing model components..."
python test_training.py  

echo "🧪 Testing ONNX export..."
python test_onnx_export.py

echo "🧪 Testing evaluation components..."
python test_evaluation.py

# echo "🧪 Testing interpretability components..."
# python test_interpretability.py

echo "🧪 Testing visualization components..."
python test_visualization.py

# Step 5: Run full integration test
echo "🔗 Step 5: Integration Test"
python test_full_pipeline.py

# Step 6: Final validation
echo "✅ Step 6: Final Validation"
./validation_checklist.sh

echo ""
echo "🎉 All tests completed!"
echo "Check the output above to see if you're ready for Novita AI deployment."