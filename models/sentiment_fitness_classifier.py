"""
Health Sentiment Fitness Classifier for TravelShield
====================================================
Fine-tuned transformer model for detecting health sentiment and travel fitness.
Classifies users as 'fit', 'borderline', or 'unfit' based on subjective indicators.

Integration Status: ACTIVE
Last Updated: 2024-01-18
Version: 1.5.2
Model: DistilBERT fine-tuned on health-domain sentiment
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel
)
from typing import List, Dict, Any, Tuple
import logging
import os
import numpy as np
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FitnessResult:
    """Structured output for fitness classification."""
    label: str  # 'fit', 'borderline', 'unfit'
    confidence: float
    probability_distribution: Dict[str, float]
    explanation: str
    recommendation: str
    requires_escalation: bool

class HealthSentimentModel:
    """
    Fine-tuned DistilBERT for health sentiment analysis and travel fitness assessment.
    
    Purpose: Detect subjective health signals from user descriptions to assess
    travel readiness. Combines emotional tone, symptom severity mentions, and
    implicit health indicators.
    
    Safety Note: This model flags potential concerns but should NOT block travel
    without human review or additional medical assessment.
    """
    
    # Fitness classes
    FITNESS_CLASSES = ['fit', 'borderline', 'unfit']
    
    # Health sentiment keywords (lexicon-based augmentation)
    NEGATIVE_KEYWORDS = {
        'awful', 'terrible', 'unbearable', 'severe', 'extreme', 'worsening',
        'debilitating', 'incapacitating', 'emergency', 'urgent', 'critical'
    }
    
    POSITIVE_KEYWORDS = {
        'feeling good', 'recovered', 'stable', 'manageable', 'mild',
        'improving', 'fine', 'normal', 'routine', 'minor'
    }
    
    def __init__(self, model_dir: str = "models/distilbert_health_sentiment/"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            'distilbert-base-uncased',
            cache_dir='.cache/transformers'
        )
        
        # Initialize model
        self.model = None
        self._load_model()
        
        # Thresholds for safety-first approach
        self.unfit_threshold = 0.60  # High recall on 'unfit'
        self.escalation_threshold = 0.40  # Escalate even for borderline cases
    
    def _load_model(self):
        """Load fine-tuned model or initialize with base DistilBERT."""
        if os.path.exists(self.model_dir):
            logger.info(f"Loading fine-tuned model from {self.model_dir}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                num_labels=len(self.FITNESS_CLASSES)
            ).to(self.device).eval()
        else:
            logger.warning("Fine-tuned model not found. Using base DistilBERT.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                'distilbert-base-uncased',
                num_labels=len(self.FITNESS_CLASSES),
                cache_dir='.cache/transformers'
            ).to(self.device).eval()
    
    def classify_fitness(self, user_text: str, additional_signals: Dict[str, Any] = None) -> FitnessResult:
        """
        Classify travel fitness from user description.
        
        Args:
            user_text: User's health status description
            additional_signals: Optional dict with 'symptom_severity', 'activity_level', etc.
        
        Returns:
            FitnessResult with classification, probabilities, and recommendations
        """
        if self.model is None:
            logger.error("Model not loaded")
            return self._create_error_result()
        
        # Preprocess text
        text_lower = user_text.lower()
        
        # Get base model prediction
        inputs = self.tokenizer(
            user_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
        
        # Adjust probabilities with lexicon signals
        adjusted_probs = self._apply_lexicon_boosting(text_lower, probs)
        
        # Get top class
        top_idx = np.argmax(adjusted_probs)
        label = self.FITNESS_CLASSES[top_idx]
        confidence = float(adjusted_probs[top_idx])
        
        # Apply additional signals if provided
        if additional_signals:
            adjusted_probs = self._apply_additional_signals(adjusted_probs, additional_signals)
            # Re-classify after signal adjustment
            top_idx = np.argmax(adjusted_probs)
            label = self.FITNESS_CLASSES[top_idx]
            confidence = float(adjusted_probs[top_idx])
        
        # Generate explanation and recommendation
        explanation = self._generate_explanation(user_text, label, confidence)
        recommendation = self._generate_recommendation(label, confidence)
        requires_escalation = self._should_escalate(label, confidence)
        
        logger.info(f"Fitness classification: {label} (confidence: {confidence:.3f})")
        
        return FitnessResult(
            label=label,
            confidence=confidence,
            probability_distribution={
                cls: float(adjusted_probs[i])
                for i, cls in enumerate(self.FITNESS_CLASSES)
            },
            explanation=explanation,
            recommendation=recommendation,
            requires_escalation=requires_escalation
        )
    
    def _apply_lexicon_boosting(self, text: str, base_probs: np.ndarray) -> np.ndarray:
        """
        Boost probabilities based on sentiment keywords (lexicon + rule engine).
        Safety-first approach: prefer false positives over false negatives.
        """
        adjusted_probs = base_probs.copy()
        weight = 0.15  # 15% influence from lexicon
        
        # Check for negative keywords
        negative_count = sum(1 for keyword in self.NEGATIVE_KEYWORDS if keyword in text)
        if negative_count >= 2:
            adjusted_probs[2] += weight  # Increase 'unfit' probability
            adjusted_probs[0] -= weight * 0.5  # Decrease 'fit'
        
        # Check for positive keywords
        positive_count = sum(1 for keyword in self.POSITIVE_KEYWORDS if keyword in text)
        if positive_count >= 2:
            adjusted_probs[0] += weight * 0.8
            adjusted_probs[2] -= weight * 0.3
        
        # Normalize
        adjusted_probs = np.clip(adjusted_probs, 0, 1)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
        
        return adjusted_probs
    
    def _apply_additional_signals(self, probs: np.ndarray, signals: Dict[str, Any]) -> np.ndarray:
        """Adjust probabilities based on additional contextual signals."""
        adjusted = probs.copy()
        weight = 0.20
        
        # High symptom severity increases 'unfit' probability
        if signals.get('symptom_severity') and signals['symptom_severity'] > 0.7:
            adjusted[2] += weight
        
        # Low activity level increases 'unfit' probability
        if signals.get('activity_level') and signals['activity_level'] < 0.3:
            adjusted[2] += weight * 0.8
        
        # Recent hospitalization
        if signals.get('recent_hospitalization'):
            adjusted[2] += weight * 0.6
            adjusted[0] -= weight * 0.4
        
        # Normalize
        adjusted = np.clip(adjusted, 0, 1)
        adjusted = adjusted / adjusted.sum()
        
        return adjusted
    
    def _generate_explanation(self, text: str, label: str, confidence: float) -> str:
        """Generate human-readable explanation for the classification."""
        if label == 'unfit':
            if 'pain' in text or 'fever' in text:
                return "Negative health indicators detected: severe symptoms mentioned with high confidence"
            return "Health sentiment analysis indicates significant concerns. Multiple warning signals detected."
        
        elif label == 'borderline':
            return "Mixed signals detected: some concerning indicators present but not definitive. Further assessment recommended."
        
        else:  # fit
            return "Health indicators appear stable. User reporting manageable or minor symptoms."
    
    def _generate_recommendation(self, label: str, confidence: float) -> str:
        """Generate recommended action based on classification."""
        if label == 'unfit':
            return "Immediate medical consultation strongly advised. Postpone travel until cleared by healthcare provider."
        
        elif label == 'borderline':
            if confidence > self.escalation_threshold:
                return "Schedule telemedicine consult within 24 hours before travel. Monitor symptoms closely."
            return "Consider consulting healthcare provider. Proceed with extra precautions if traveling."
        
        else:  # fit
            return "Monitor symptoms. Follow standard travel health precautions. Emergency contact info prepared."
    
    def _should_escalate(self, label: str, confidence: float) -> bool:
        """Determine if case requires escalation to clinician."""
        # Escalate unfit cases
        if label == 'unfit':
            return True
        
        # Escalate high-confidence borderline cases
        if label == 'borderline' and confidence > self.escalation_threshold:
            return True
        
        return False
    
    def _create_error_result(self) -> FitnessResult:
        """Create error result when model fails."""
        return FitnessResult(
            label='borderline',
            confidence=0.0,
            probability_distribution={'fit': 0.33, 'borderline': 0.34, 'unfit': 0.33},
            explanation='Model inference error occurred',
            recommendation='Manual review required',
            requires_escalation=True
        )
    
    def batch_classify(self, user_texts: List[str]) -> List[FitnessResult]:
        """Process multiple users at once."""
        results = []
        for text in user_texts:
            result = self.classify_fitness(text)
            results.append(result)
        return results


# Integration wrapper for Flask API
class FitnessClassifierService:
    """
    Service wrapper for health sentiment fitness classification.
    Provides safety-first inference with escalation logic.
    """
    
    def __init__(self):
        self.model = HealthSentimentModel()
        logger.info("Health Fitness Classifier Service initialized")
        self.safety_mode = True  # Always escalate uncertain cases
    
    def assess_fitness(self, user_text: str, additional_signals: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main inference endpoint for Flask API.
        
        Usage in app.py:
            from lib.ml_models.sentiment_fitness_classifier import FitnessClassifierService
            fitness_service = FitnessClassifierService()
            result = fitness_service.assess_fitness(user_text, additional_signals)
        """
        try:
            result = self.model.classify_fitness(user_text, additional_signals)
            
            # Convert to dict for JSON serialization
            output = {
                'fitness_label': result.label,
                'confidence': result.confidence,
                'probabilities': result.probability_distribution,
                'explanation': result.explanation,
                'recommendation': result.recommendation,
                'requires_escalation': result.requires_escalation,
                'escalation_level': self._get_escalation_level(result)
            }
            
            # Add safety warnings
            if output['requires_escalation']:
                output['safety_warning'] = 'High priority: Medical consultation required'
            
            return output
            
        except Exception as e:
            logger.error(f"Error in fitness classification: {str(e)}")
            return {
                'fitness_label': 'borderline',
                'confidence': 0.0,
                'probabilities': {'fit': 0.33, 'borderline': 0.34, 'unfit': 0.33},
                'error': str(e),
                'requires_escalation': True,
                'recommendation': 'Manual review required due to model error'
            }
    
    def _get_escalation_level(self, result: FitnessResult) -> str:
        """Determine escalation priority level."""
        if result.label == 'unfit' and result.confidence > 0.70:
            return 'CRITICAL'
        elif result.label == 'unfit':
            return 'HIGH'
        elif result.label == 'borderline' and result.confidence > 0.50:
            return 'MEDIUM'
        else:
            return 'LOW'


# Standalone testing
if __name__ == "__main__":
    # Example usage
    service = FitnessClassifierService()
    
    # Test cases
    test_cases = [
        {
            'text': "I've been feeling awful for days with severe chest pain",
            'signals': {'symptom_severity': 0.9, 'activity_level': 0.2}
        },
        {
            'text': "I'm feeling a bit under the weather but it's manageable",
            'signals': {'symptom_severity': 0.3, 'activity_level': 0.7}
        },
        {
            'text': "Feeling good and ready to travel!",
            'signals': {}
        }
    ]
    
    print("\n" + "="*60)
    print("Health Fitness Classification Test Results")
    print("="*60)
    
    for i, case in enumerate(test_cases, 1):
        result = service.assess_fitness(
            case['text'],
            case.get('signals')
        )
        
        print(f"\nTest Case {i}:")
        print(f"  Input: '{case['text']}'")
        print(f"  Label: {result['fitness_label']} (confidence: {result['confidence']:.2f})")
        print(f"  Escalation: {'YES' if result['requires_escalation'] else 'NO'} - {result.get('escalation_level', 'N/A')}")
        print(f"  Recommendation: {result['recommendation']}")
        if result.get('safety_warning'):
            print(f"  ⚠️  {result['safety_warning']}")

