"""
Clinical Explanation Generator for HMS EEG Classification System.
Provides clinically meaningful explanations for model predictions.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from . import ExplanationResult


@dataclass
class ClinicalExplanation:
    """Container for clinical explanations."""
    prediction: str
    confidence: float
    key_features: List[str]
    clinical_evidence: Dict[str, Any]
    risk_factors: List[str]
    recommendations: List[str]
    explanation_text: str
    
    
class ClinicalExplanationGenerator:
    """Generate clinically meaningful explanations for EEG predictions."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Initialize clinical explanation generator.
        
        Args:
            model: Trained model
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Clinical mappings
        self.class_names = config.get('classes', [
            'seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other'
        ])
        
        # Clinical knowledge base
        self.clinical_knowledge = self._load_clinical_knowledge()
        
    def _load_clinical_knowledge(self) -> Dict[str, Dict]:
        """Load clinical knowledge about different brain activity patterns."""
        return {
            'seizure': {
                'description': 'Abnormal electrical activity in the brain causing seizures',
                'characteristics': [
                    'Rhythmic or spike patterns',
                    'Evolution in frequency/amplitude',
                    'Focal or generalized onset'
                ],
                'clinical_significance': 'Requires immediate medical attention',
                'typical_features': ['sharp waves', 'spikes', 'rhythmic activity'],
                'frequency_bands': ['delta', 'theta', 'alpha', 'beta'],
                'recommendations': [
                    'Confirm with neurologist',
                    'Consider antiepileptic medications',
                    'Monitor for additional seizures'
                ]
            },
            'lpd': {
                'description': 'Lateralized periodic discharges',
                'characteristics': [
                    'Periodic sharp waves',
                    'Lateralized to one hemisphere',
                    'Regular interval patterns'
                ],
                'clinical_significance': 'Associated with acute brain injury',
                'typical_features': ['periodic discharges', 'lateralization'],
                'frequency_bands': ['delta', 'theta'],
                'recommendations': [
                    'Assess for underlying brain pathology',
                    'Consider EEG monitoring',
                    'Evaluate need for treatment'
                ]
            },
            'gpd': {
                'description': 'Generalized periodic discharges',
                'characteristics': [
                    'Bilateral periodic patterns',
                    'Generalized distribution',
                    'Regular intervals'
                ],
                'clinical_significance': 'May indicate diffuse brain dysfunction',
                'typical_features': ['bilateral discharges', 'generalized pattern'],
                'frequency_bands': ['delta', 'theta'],
                'recommendations': [
                    'Evaluate for metabolic causes',
                    'Consider neuroimaging',
                    'Monitor clinical status'
                ]
            },
            'lrda': {
                'description': 'Lateralized rhythmic delta activity',
                'characteristics': [
                    'Rhythmic delta waves',
                    'Lateralized pattern',
                    'Continuous activity'
                ],
                'clinical_significance': 'May suggest focal brain dysfunction',
                'typical_features': ['rhythmic delta', 'lateralization'],
                'frequency_bands': ['delta'],
                'recommendations': [
                    'Investigate focal pathology',
                    'Consider neuroimaging',
                    'Clinical correlation needed'
                ]
            },
            'grda': {
                'description': 'Generalized rhythmic delta activity',
                'characteristics': [
                    'Bilateral rhythmic delta',
                    'Generalized distribution',
                    'Continuous pattern'
                ],
                'clinical_significance': 'May indicate encephalopathy',
                'typical_features': ['rhythmic delta', 'generalized'],
                'frequency_bands': ['delta'],
                'recommendations': [
                    'Evaluate for encephalopathy',
                    'Check metabolic parameters',
                    'Consider treatment of underlying cause'
                ]
            },
            'other': {
                'description': 'Other patterns or normal EEG activity',
                'characteristics': [
                    'Normal background activity',
                    'Non-specific changes',
                    'Artifact or unclear patterns'
                ],
                'clinical_significance': 'Variable clinical significance',
                'typical_features': ['normal variants', 'artifacts'],
                'frequency_bands': ['alpha', 'beta'],
                'recommendations': [
                    'Continue monitoring if indicated',
                    'Clinical correlation',
                    'Consider repeat EEG if needed'
                ]
            }
        }
        
    def generate_explanation(self, 
                           x: torch.Tensor,
                           prediction_result: Dict[str, Any],
                           patient_context: Optional[Dict[str, Any]] = None) -> ClinicalExplanation:
        """
        Generate clinical explanation for a prediction.
        
        Args:
            x: Input EEG data
            prediction_result: Model prediction results
            patient_context: Optional patient clinical context
            
        Returns:
            ClinicalExplanation object
        """
        # Extract prediction details
        if isinstance(prediction_result, dict):
            logits = prediction_result.get('logits', prediction_result.get('predictions'))
            if isinstance(logits, torch.Tensor):
                logits = logits.cpu().numpy()
            probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        else:
            probabilities = prediction_result
            
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])
        
        # Get clinical knowledge for predicted class
        clinical_info = self.clinical_knowledge.get(predicted_class, {})
        
        # Analyze key features (simplified)
        key_features = self._identify_key_features(x, predicted_class)
        
        # Generate clinical evidence
        clinical_evidence = self._analyze_clinical_evidence(x, predicted_class, patient_context)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(predicted_class, patient_context)
        
        # Generate recommendations
        recommendations = clinical_info.get('recommendations', [])
        if patient_context:
            recommendations.extend(self._get_context_specific_recommendations(
                predicted_class, patient_context
            ))
        
        # Generate explanation text
        explanation_text = self._generate_explanation_text(
            predicted_class, confidence, key_features, clinical_evidence
        )
        
        return ClinicalExplanation(
            prediction=predicted_class,
            confidence=confidence,
            key_features=key_features,
            clinical_evidence=clinical_evidence,
            risk_factors=risk_factors,
            recommendations=recommendations,
            explanation_text=explanation_text
        )
        
    def _identify_key_features(self, x: torch.Tensor, predicted_class: str) -> List[str]:
        """Identify key EEG features contributing to the prediction."""
        # Simplified feature analysis
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        
        key_features = []
        
        # Analyze frequency content (simplified)
        if x_np.ndim >= 2:
            # Check for high amplitude in different frequency ranges
            mean_amplitude = np.mean(np.abs(x_np), axis=-1)
            
            if np.max(mean_amplitude) > np.mean(mean_amplitude) * 2:
                key_features.append("High amplitude activity detected")
                
            # Check for rhythmic patterns (simplified)
            if predicted_class in ['lrda', 'grda']:
                key_features.append("Rhythmic delta activity pattern")
            elif predicted_class in ['lpd', 'gpd']:
                key_features.append("Periodic discharge pattern")
            elif predicted_class == 'seizure':
                key_features.append("Ictal pattern with evolution")
                
        return key_features
        
    def _analyze_clinical_evidence(self, 
                                 x: torch.Tensor, 
                                 predicted_class: str,
                                 patient_context: Optional[Dict]) -> Dict[str, Any]:
        """Analyze clinical evidence supporting the prediction."""
        evidence = {
            'pattern_type': predicted_class,
            'signal_characteristics': {},
            'distribution': 'unknown',
            'frequency_content': {}
        }
        
        # Basic signal analysis
        x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
        
        if x_np.ndim >= 2:
            evidence['signal_characteristics'] = {
                'mean_amplitude': float(np.mean(np.abs(x_np))),
                'max_amplitude': float(np.max(np.abs(x_np))),
                'variance': float(np.var(x_np))
            }
            
            # Determine distribution pattern
            if predicted_class.startswith('l'):  # lateralized
                evidence['distribution'] = 'lateralized'
            elif predicted_class.startswith('g'):  # generalized
                evidence['distribution'] = 'generalized'
            elif predicted_class == 'seizure':
                evidence['distribution'] = 'focal_or_generalized'
                
        return evidence
        
    def _identify_risk_factors(self, 
                             predicted_class: str,
                             patient_context: Optional[Dict]) -> List[str]:
        """Identify clinical risk factors."""
        risk_factors = []
        
        # General risk factors based on pattern
        if predicted_class == 'seizure':
            risk_factors.extend([
                'History of epilepsy',
                'Brain injury or trauma',
                'Metabolic disturbances',
                'Medication non-compliance'
            ])
        elif predicted_class in ['lpd', 'gpd']:
            risk_factors.extend([
                'Acute brain injury',
                'Stroke or hemorrhage',
                'Encephalitis',
                'Metabolic encephalopathy'
            ])
        elif predicted_class in ['lrda', 'grda']:
            risk_factors.extend([
                'Encephalopathy',
                'Metabolic disorders',
                'Toxic exposures',
                'Hypoxic brain injury'
            ])
            
        # Add patient-specific risk factors
        if patient_context:
            age = patient_context.get('age')
            if age and age > 65:
                risk_factors.append('Advanced age')
                
            medical_history = patient_context.get('medical_history', [])
            for condition in medical_history:
                if condition.lower() in ['stroke', 'tbi', 'epilepsy']:
                    risk_factors.append(f'History of {condition}')
                    
        return risk_factors
        
    def _get_context_specific_recommendations(self,
                                            predicted_class: str,
                                            patient_context: Dict) -> List[str]:
        """Get recommendations specific to patient context."""
        recommendations = []
        
        age = patient_context.get('age')
        if age and age < 18:
            recommendations.append('Consider pediatric neurology consultation')
        elif age and age > 65:
            recommendations.append('Evaluate for age-related factors')
            
        medications = patient_context.get('medications', [])
        if any('anticonvulsant' in med.lower() for med in medications):
            recommendations.append('Review current anticonvulsant levels')
            
        return recommendations
        
    def _generate_explanation_text(self,
                                 predicted_class: str,
                                 confidence: float,
                                 key_features: List[str],
                                 clinical_evidence: Dict) -> str:
        """Generate human-readable explanation text."""
        clinical_info = self.clinical_knowledge.get(predicted_class, {})
        
        explanation = f"The EEG pattern is classified as '{predicted_class}' with {confidence:.1%} confidence.\n\n"
        
        # Add description
        description = clinical_info.get('description', 'Unknown pattern')
        explanation += f"Description: {description}\n\n"
        
        # Add key features
        if key_features:
            explanation += "Key identifying features:\n"
            for feature in key_features:
                explanation += f"• {feature}\n"
            explanation += "\n"
            
        # Add clinical significance
        significance = clinical_info.get('clinical_significance', 'Clinical significance unknown')
        explanation += f"Clinical significance: {significance}\n\n"
        
        # Add confidence interpretation
        if confidence > 0.8:
            explanation += "High confidence prediction - pattern characteristics are clear and well-defined."
        elif confidence > 0.6:
            explanation += "Moderate confidence prediction - some pattern characteristics present."
        else:
            explanation += "Low confidence prediction - pattern may be ambiguous or require additional evaluation."
            
        return explanation
        
    def generate_report(self, 
                       explanation: ClinicalExplanation,
                       save_path: Optional[str] = None) -> str:
        """Generate a formatted clinical report."""
        report = f"""
EEG CLASSIFICATION REPORT
========================

PREDICTION: {explanation.prediction.upper()}
CONFIDENCE: {explanation.confidence:.1%}

CLINICAL FINDINGS:
{explanation.explanation_text}

KEY FEATURES:
{chr(10).join(f"• {feature}" for feature in explanation.key_features)}

RISK FACTORS:
{chr(10).join(f"• {factor}" for factor in explanation.risk_factors)}

RECOMMENDATIONS:
{chr(10).join(f"• {rec}" for rec in explanation.recommendations)}

CLINICAL EVIDENCE:
Pattern Type: {explanation.clinical_evidence.get('pattern_type', 'Unknown')}
Distribution: {explanation.clinical_evidence.get('distribution', 'Unknown')}

Signal Characteristics:
{chr(10).join(f"  {k}: {v}" for k, v in explanation.clinical_evidence.get('signal_characteristics', {}).items())}

---
Report generated by HMS EEG Classification System
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report
        
    def visualize_explanation(self,
                            explanation: ClinicalExplanation,
                            save_path: Optional[str] = None):
        """Create visual explanation of the prediction."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Confidence bar
        axes[0, 0].bar(['Confidence'], [explanation.confidence], color='skyblue')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title('Prediction Confidence')
        axes[0, 0].set_ylabel('Confidence')
        
        # Key features
        if explanation.key_features:
            axes[0, 1].barh(range(len(explanation.key_features)), 
                           [1] * len(explanation.key_features), color='lightgreen')
            axes[0, 1].set_yticks(range(len(explanation.key_features)))
            axes[0, 1].set_yticklabels(explanation.key_features)
            axes[0, 1].set_title('Key Features')
        
        # Risk factors
        if explanation.risk_factors:
            axes[1, 0].barh(range(len(explanation.risk_factors)), 
                           [1] * len(explanation.risk_factors), color='lightcoral')
            axes[1, 0].set_yticks(range(len(explanation.risk_factors)))
            axes[1, 0].set_yticklabels(explanation.risk_factors)
            axes[1, 0].set_title('Risk Factors')
        
        # Recommendations
        if explanation.recommendations:
            axes[1, 1].barh(range(len(explanation.recommendations)), 
                           [1] * len(explanation.recommendations), color='lightyellow')
            axes[1, 1].set_yticks(range(len(explanation.recommendations)))
            axes[1, 1].set_yticklabels(explanation.recommendations)
            axes[1, 1].set_title('Recommendations')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close() 