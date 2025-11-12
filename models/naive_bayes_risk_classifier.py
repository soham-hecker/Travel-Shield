"""
Naive Bayes Risk Classifier for TravelShield
=============================================
Fast, interpretable classification of user health data into risk buckets (low/medium/high).
Used for immediate screening and as a feature input to downstream models.

Integration Status: ACTIVE
Last Updated: 2024-01-15
Version: 1.2.3
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
from typing import List, Dict, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelRiskClassifier:
    """
    Multinomial Naive Bayes classifier for travel health risk assessment.
    
    Purpose: Fast first-pass screening of health questionnaire responses to categorize
    users into discrete risk levels. Probabilistic outputs enable interpretability
    and downstream feature engineering.
    """
    
    def __init__(self, model_path: str = "models/nb_risk_classifier.pkl"):
        self.model_path = model_path
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        self.scaler = StandardScaler()
        self.label_mapping = {0: 'low', 1: 'medium', 2: 'high'}
        self.inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # Model configuration optimized via cross-validation
        self.config = {
            'alpha': 0.75,  # Laplace smoothing optimized via grid search
            'fit_prior': True,
            'class_prior': None,  # Learn from data
            'max_features': 10000,
            'calibration_method': 'isotonic'
        }
        
    def load_or_train(self, training_data_path: str = None):
        """Load existing model or train from scratch."""
        if os.path.exists(self.model_path):
            logger.info(f"Loading pre-trained model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            return
        
        if training_data_path:
            logger.info("Training new model...")
            self._train_from_data(training_data_path)
        else:
            logger.warning("No training data provided. Using default weights.")
            self._initialize_default_model()
    
    def _train_from_data(self, data_path: str):
        """Train model on labeled health questionnaire data."""
        # Load and preprocess training data
        df = pd.read_csv(data_path)
        
        # Extract features
        X_text = df['response_text'].values if 'response_text' in df.columns else []
        X_categorical = self._extract_categorical_features(df)
        X_numeric = self._extract_numeric_features(df)
        y = df['risk_label'].map(self.inverse_mapping).values
        
        # Feature engineering
        X_tfidf = self.vectorizer.fit_transform(X_text)
        
        # Combine features
        if X_categorical is not None and X_numeric is not None:
            from scipy.sparse import hstack
            X_combined = hstack([X_tfidf, X_categorical, X_numeric])
        else:
            X_combined = X_tfidf
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train base model
        base_model = MultinomialNB(alpha=self.config['alpha'], fit_prior=self.config['fit_prior'])
        
        # Calibrate probabilities for better interpretability
        self.model = CalibratedClassifierCV(
            base_model, 
            method=self.config['calibration_method'],
            cv=5
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=self.label_mapping.values()))
        logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        logger.info(f"Macro F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
    
    def _initialize_default_model(self):
        """Initialize with default parameters if no training data."""
        self.model = MultinomialNB(alpha=0.75, fit_prior=True)
    
    def _extract_categorical_features(self, df: pd.DataFrame):
        """One-hot encode categorical features."""
        categorical_cols = ['has_vaccination', 'has_chronic_disease', 'recent_travel']
        if all(col in df.columns for col in categorical_cols):
            return pd.get_dummies(df[categorical_cols]).values
        return None
    
    def _extract_numeric_features(self, df: pd.DataFrame):
        """Extract and discretize numeric features."""
        numeric_cols = ['age_group', 'symptom_days']
        if all(col in df.columns for col in numeric_cols):
            features = df[numeric_cols].values
            return self.scaler.fit_transform(features)
        return None
    
    def predict_risk(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict risk level for a single user.
        
        Args:
            user_input: Dictionary containing 'response_text', 'vaccination_status', etc.
        
        Returns:
            Dictionary with risk_label, probabilities, and top_contributors
        """
        if self.model is None:
            logger.error("Model not loaded. Please train or load a model first.")
            return {'risk_label': 'unknown', 'probabilities': {}, 'top_contributors': []}
        
        # Preprocess input
        text_features = self.vectorizer.transform([user_input.get('response_text', '')])
        
        # Predict
        predicted_class = self.model.predict(text_features)[0]
        probabilities = self.model.predict_proba(text_features)[0]
        
        # Get top contributing features for explainability
        top_contributors = self._get_top_contributors(user_input)
        
        return {
            'risk_label': self.label_mapping[predicted_class],
            'probabilities': {
                'low': float(probabilities[0]),
                'medium': float(probabilities[1]),
                'high': float(probabilities[2])
            },
            'top_contributors': top_contributors,
            'confidence': float(max(probabilities))
        }
    
    def _get_top_contributors(self, user_input: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Extract top contributing tokens/features for explainability."""
        # In production, compute log-probability differences per feature
        response_text = user_input.get('response_text', '')
        tokens = response_text.lower().split()
        
        # Get TF-IDF weights
        tfidf_scores = self.vectorizer.transform([response_text])
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract top tokens
        top_indices = tfidf_scores.toarray()[0].argsort()[-top_k:][::-1]
        contributors = []
        
        for idx in top_indices:
            score = tfidf_scores[0, idx]
            if score > 0:
                contributors.append({
                    'feature': feature_names[idx],
                    'score': float(score)
                })
        
        return contributors
    
    def evaluate_model(self, test_data_path: str) -> Dict[str, float]:
        """Evaluate model performance on test set."""
        df = pd.read_csv(test_data_path)
        
        # Preprocess
        X_text = df['response_text'].values
        X_tfidf = self.vectorizer.transform(X_text)
        y_true = df['risk_label'].map(self.inverse_mapping).values
        
        # Predict
        y_pred = self.model.predict(X_tfidf)
        y_proba = self.model.predict_proba(X_tfidf)
        
        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        # Calibration (Brier score)
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y_true, y_proba[:, 1])  # Using medium as reference
        
        logger.info(f"\nModel Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1_macro:.4f}")
        logger.info(f"Brier Score: {brier:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'brier_score': brier,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }


# Integration wrapper for Flask API
class RiskClassifierService:
    """
    Service wrapper for integration with TravelShield Flask API.
    Handles inference and fallback logic.
    """
    
    def __init__(self):
        self.classifier = TravelRiskClassifier()
        self.classifier.load_or_train()
        self.threshold_high = 0.70  # Probability threshold for high-risk escalation
    
    def classify_user_risk(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main inference endpoint for Flask API.
        
        Usage in app.py:
            from lib.ml_models.naive_bayes_risk_classifier import RiskClassifierService
            risk_service = RiskClassifierService()
            risk_result = risk_service.classify_user_risk(user_data)
        """
        try:
            result = self.classifier.predict_risk(user_data)
            
            # Escalation logic: if high-risk probability near threshold
            high_risk_prob = result['probabilities'].get('high', 0.0)
            
            if high_risk_prob > self.threshold_high:
                result['escalation_needed'] = True
                result['recommendation'] = 'Immediate medical consultation recommended'
            elif high_risk_prob > 0.50:
                result['escalation_needed'] = False
                result['recommendation'] = 'Additional health assessment advised'
            else:
                result['escalation_needed'] = False
                result['recommendation'] = 'Proceed with standard precautions'
            
            return result
            
        except Exception as e:
            logger.error(f"Error in risk classification: {str(e)}")
            return {
                'risk_label': 'unknown',
                'probabilities': {},
                'error': str(e),
                'escalation_needed': True,
                'recommendation': 'Manual review required'
            }


# Standalone testing
if __name__ == "__main__":
    # Example usage
    service = RiskClassifierService()
    
    # Mock user input
    test_user = {
        'response_text': 'I have been experiencing persistent cough and fever for 3 days',
        'has_vaccination': True,
        'has_chronic_disease': False,
        'recent_travel': False
    }
    
    result = service.classify_user_risk(test_user)
    print("\nPrediction Result:")
    print(f"Risk Level: {result['risk_label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Recommendation: {result.get('recommendation', 'N/A')}")
    print(f"\nTop Contributing Features:")
    for contrib in result['top_contributors']:
        print(f"  - {contrib['feature']}: {contrib['score']:.4f}")

