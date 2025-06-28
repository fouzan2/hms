"""
Comprehensive evaluation and validation module for EEG harmful brain activity classification.
"""

# Existing evaluator
from .evaluator import ModelEvaluator, ClinicalMetrics, BiasDetector, EvaluationVisualizer

# Performance metrics
from .performance_metrics import (
    PerformanceMetrics,
    TemporalMetrics,
    MetricResult
)

# Cross-validation strategies
from .cross_validation import (
    CrossValidationStrategy,
    StandardKFold,
    StratifiedKFoldCV,
    PatientWiseCV,
    StratifiedPatientCV,
    TimeBasedCV,
    BlockedTimeSeriesCV,
    NestedCV,
    MonteCarloCV,
    LeaveOnePatientOut,
    CrossValidationManager,
    CVFold
)

# Robustness testing
from .robustness_testing import (
    NoiseRobustnessTester,
    AdversarialRobustnessTester,
    DistributionShiftTester,
    CalibrationTester,
    ConsistencyTester,
    RobustnessReporter,
    RobustnessResult
)

# Clinical validation
from .clinical_validation import (
    ClinicalAgreementValidator,
    ClinicalTrialSimulator,
    ClinicalPerformanceAnalyzer,
    ClinicalReportGenerator,
    ClinicalValidationResult
)

__all__ = [
    # Existing
    'ModelEvaluator',
    'ClinicalMetrics',
    'BiasDetector',
    'EvaluationVisualizer',
    
    # Performance metrics
    'PerformanceMetrics',
    'TemporalMetrics',
    'MetricResult',
    
    # Cross-validation
    'CrossValidationStrategy',
    'StandardKFold',
    'StratifiedKFoldCV',
    'PatientWiseCV',
    'StratifiedPatientCV',
    'TimeBasedCV',
    'BlockedTimeSeriesCV',
    'NestedCV',
    'MonteCarloCV',
    'LeaveOnePatientOut',
    'CrossValidationManager',
    'CVFold',
    
    # Robustness testing
    'NoiseRobustnessTester',
    'AdversarialRobustnessTester',
    'DistributionShiftTester',
    'CalibrationTester',
    'ConsistencyTester',
    'RobustnessReporter',
    'RobustnessResult',
    
    # Clinical validation
    'ClinicalAgreementValidator',
    'ClinicalTrialSimulator',
    'ClinicalPerformanceAnalyzer',
    'ClinicalReportGenerator',
    'ClinicalValidationResult'
] 