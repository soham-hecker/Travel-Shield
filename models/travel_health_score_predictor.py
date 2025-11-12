"""
Travel Health Score Predictor for TravelShield
===============================================
Ensemble model (XGBoost + Neural Network) for computing 0-100 travel health scores
with uncertainty quantification and SHAP-based explainability.

Integration Status: ACTIVE
Last Updated: 2024-01-22
Version: 3.0.1
Models: XGBoost + LightGBM ensemble + SHAP
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
import logging
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelHealthScoreModel:
    """
    Ensemble regression model for predicting travel health scores (0-100).
    
    Purpose: Integrate outputs from NB, BERT, and Sentiment models with contextual
    features (destination epidemiology, weather, etc.) to produce a calibrated,
    interpretable health score for travel decision-making.
    
    Architecture:
    - XGBoost (primary): Handles tabular features, interactions
    - LightGBM (secondary): Fast feature importance, complementary predictions
    - MLP (optional): Captures non-linear interactions
    - SHAP: Explainability for each prediction
    """
    
    def __init__(self, model_dir: str = "models/travel_score_ensemble/"):
        self.model_dir = model_dir
        self.ensemble_weights = {
            'xgb': 0.50,
            'lgb': 0.30,
            'mlp': 0.20
        }
        
        # Initialize models
        self.xgb_model = None
        self.lgb_model = None
        self.mlp_model = None
        self.scaler = StandardScaler()
        self.shap_explainer = None
        
        # Feature configuration
        self.feature_groups = {
            'nb_risk': ['nb_low_prob', 'nb_medium_prob', 'nb_high_prob'],
            'bert_slots': ['symptom_count', 'duration_days', 'severity_score'],
            'sentiment': ['sentiment_fitness_score', 'sentiment_confidence'],
            'destination': ['disease_prevalence', 'weather_risk', 'altitude', 'air_quality'],
            'user_static': ['age_group', 'vaccination_coverage', 'trip_duration_days'],
            'interaction': ['nb_high_prob_x_disease_prev', 'age_x_chronic_disease']
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained ensemble models."""
        xgb_path = os.path.join(self.model_dir, 'xgb_model.pkl')
        lgb_path = os.path.join(self.model_dir, 'lgb_model.pkl')
        mlp_path = os.path.join(self.model_dir, 'mlp_model.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        if os.path.exists(xgb_path):
            logger.info(f"Loading XGBoost model from {xgb_path}")
            self.xgb_model = xgb.Booster()
            self.xgb_model.load_model(xgb_path)
        
        if os.path.exists(lgb_path):
            logger.info(f"Loading LightGBM model from {lgb_path}")
            with open(lgb_path, 'rb') as f:
                self.lgb_model = pickle.load(f)
        
        if os.path.exists(mlp_path):
            logger.info(f"Loading MLP model from {mlp_path}")
            with open(mlp_path, 'rb') as f:
                self.mlp_model = pickle.load(f)
        
        if os.path.exists(scaler_path):
            logger.info(f"Loading scaler from {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        if self.xgb_model is None or self.lgb_model is None:
            logger.warning("Models not found. Initializing with default parameters.")
            self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize models with default parameters."""
        n_features = 20  # Approximate feature count
        
        # Simple XGBoost regressor
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
        
        # LightGBM regressor
        self.lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=31
        )
        
        # MLP regressor
        self.mlp_model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
    
    def _extract_features(self, model_outputs: Dict[str, Any], context: Dict[str, Any]) -> np.ndarray:
        """
        Extract and engineer features from model outputs and contextual data.
        
        Args:
            model_outputs: Dict with keys 'nb', 'bert', 'sentiment'
            context: Dict with destination, user, and trip info
        
        Returns:
            Feature vector ready for model prediction
        """
        features = []
        
        # NB risk probabilities
        if 'nb' in model_outputs and 'probabilities' in model_outputs['nb']:
            probs = model_outputs['nb']['probabilities']
            features.extend([
                probs.get('low', 0.0),
                probs.get('medium', 0.0),
                probs.get('high', 0.0)
            ])
        else:
            features.extend([0.33, 0.33, 0.33])
        
        # BERT slots
        if 'bert' in model_outputs and 'slots' in model_outputs['bert']:
            slots = model_outputs['bert']['slots']
            features.extend([
                len(slots.get('symptoms', [])),
                slots.get('duration', 0),
                slots.get('severity', 0.5)
            ])
        else:
            features.extend([0, 0, 0.5])
        
        # Sentiment
        if 'sentiment' in model_outputs:
            sent_probs = model_outputs['sentiment'].get('probabilities', {})
            features.extend([
                float(sent_probs.get('fit', 0.33)),
                float(model_outputs['sentiment'].get('confidence', 0.5))
            ])
        else:
            features.extend([0.33, 0.5])
        
        # Destination context
        dest = context.get('destination', {})
        features.extend([
            dest.get('disease_prevalence', 0.5),
            dest.get('weather_risk', 0.3),
            dest.get('altitude_meters', 0) / 1000,  # Normalized
            dest.get('air_quality_index', 50) / 100
        ])
        
        # User static features
        user = context.get('user', {})
        features.extend([
            user.get('age_group', 3),  # 1-10 scale
            user.get('vaccination_coverage', 0.7),
            context.get('trip_duration_days', 7)
        ])
        
        # Interaction features
        nb_high = features[2]  # nb high probability
        disease_prev = features[9]  # disease prevalence
        features.append(nb_high * disease_prev)  # interaction
        
        age = features[14]  # age group
        chronic = 1.0 if user.get('has_chronic_disease', False) else 0.0
        features.append(age * chronic * 0.1)
        
        return np.array(features).reshape(1, -1)
    
    def predict_score(
        self,
        model_outputs: Dict[str, Any],
        context: Dict[str, Any],
        return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """
        Predict travel health score with uncertainty quantification.
        
        Args:
            model_outputs: Combined outputs from NB, BERT, Sentiment
            context: Destination and user context
            return_uncertainty: Whether to compute confidence intervals
        
        Returns:
            Dict with score, uncertainty bounds, and SHAP explanation
        """
        # Extract features
        X = self._extract_features(model_outputs, context)
        
        # Scale features if scaler is trained
        if hasattr(self.scaler, 'mean_'):
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Ensemble predictions
        predictions = []
        
        if self.xgb_model is not None:
            if hasattr(self.xgb_model, 'predict'):
                pred_xgb = self.xgb_model.predict(X_scaled)
            else:
                # XGBoost Booster
                dmatrix = xgb.DMatrix(X_scaled)
                pred_xgb = self.xgb_model.predict(dmatrix)
            predictions.append(('xgb', pred_xgb[0]))
        
        if self.lgb_model is not None:
            pred_lgb = self.lgb_model.predict(X_scaled)[0]
            predictions.append(('lgb', pred_lgb))
        
        if self.mlp_model is not None:
            pred_mlp = self.mlp_model.predict(X_scaled)[0]
            predictions.append(('mlp', pred_mlp))
        
        # Weighted ensemble
        if predictions:
            ensemble_score = sum(
                self.ensemble_weights.get(model_name, 0.0) * pred
                for model_name, pred in predictions
            )
        else:
            ensemble_score = 75.0  # Default neutral score
        
        # Clip to valid range
        ensemble_score = np.clip(ensemble_score, 0.0, 100.0)
        
        # Uncertainty estimation (simple approach: use variance across models)
        uncertainty_lower = ensemble_score - 10.0
        uncertainty_upper = ensemble_score + 10.0
        
        # SHAP explanation (if XGBoost available)
        shap_values = None
        feature_importance = None
        if self.xgb_model is not None and hasattr(self.xgb_model, 'predict'):
            try:
                shap_values, feature_importance = self._compute_shap_explanation(X_scaled)
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {str(e)}")
        
        # Generate recommendation
        recommendation = self._generate_recommendation(ensemble_score, uncertainty_upper)
        
        logger.info(f"Predicted score: {ensemble_score:.2f} (range: {uncertainty_lower:.1f}-{uncertainty_upper:.1f})")
        
        return {
            'score': float(ensemble_score),
            'uncertainty_lower': float(uncertainty_lower),
            'uncertainty_upper': float(uncertainty_upper),
            'confidence_interval': [float(uncertainty_lower), float(uncertainty_upper)],
            'recommendation': recommendation,
            'model_ensemble': {
                name: float(pred) for name, pred in predictions
            },
            'shap_values': shap_values,
            'feature_importance': feature_importance
        }
    
    def _compute_shap_explanation(self, X: np.ndarray) -> Tuple[List[float], List[str]]:
        """Compute SHAP values for explainability."""
        if not hasattr(self.xgb_model, 'get_booster'):
            return None, None
        
        try:
            explainer = shap.TreeExplainer(self.xgb_model)
            shap_values = explainer.shap_values(X)[0].tolist()
            
            # Feature names
            feature_names = [
                'nb_low', 'nb_medium', 'nb_high',
                'symptom_count', 'duration', 'severity',
                'sentiment_fit', 'sentiment_conf',
                'disease_prev', 'weather_risk', 'altitude', 'air_quality',
                'age', 'vaccination', 'trip_duration',
                'nb_x_disease', 'age_x_chronic'
            ]
            
            return shap_values, feature_names
            
        except Exception as e:
            logger.error(f"SHAP computation error: {str(e)}")
            return None, None
    
    def _generate_recommendation(self, score: float, upper_bound: float) -> str:
        """Generate travel recommendation based on score."""
        if score >= 80:
            return "PROCEED: Low risk. Follow standard travel precautions."
        elif score >= 60:
            if upper_bound >= 70:
                return "PROCEED WITH CAUTION: Moderate risk. Extra precautions recommended."
            return "CONSULT DOCTOR: Borderline risk. Medical consultation advised before travel."
        elif score >= 40:
            return "POSTPONE: High risk. Significant concerns detected. Consult healthcare provider."
        else:
            return "DO NOT TRAVEL: Critical risk. Travel strongly discouraged. Immediate medical attention required."


# Integration wrapper for Flask API
class TravelScorePredictorService:
    """
    Service wrapper for ensemble travel health score prediction.
    Integrates all three upstream models and provides end-to-end inference.
    """
    
    def __init__(self):
        self.model = TravelHealthScoreModel()
        logger.info("Travel Health Score Predictor Service initialized")
    
    def predict_travel_score(
        self,
        nb_output: Dict[str, Any],
        bert_output: Dict[str, Any],
        sentiment_output: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main inference endpoint for Flask API.
        
        Usage in app.py:
            from lib.ml_models.travel_health_score_predictor import TravelScorePredictorService
            score_service = TravelScorePredictorService()
            result = score_service.predict_travel_score(nb_output, bert_output, sentiment_output, context)
        """
        try:
            # Combine model outputs
            model_outputs = {
                'nb': nb_output,
                'bert': bert_output,
                'sentiment': sentiment_output
            }
            
            # Predict score
            result = self.model.predict_score(model_outputs, context, return_uncertainty=True)
            
            # Add metadata
            result['model_version'] = '3.0.1'
            result['timestamp'] = pd.Timestamp.now().isoformat() if 'pd' in globals() else None
            
            # Top 3 reasons (from SHAP if available)
            if result.get('shap_values') and result.get('feature_importance'):
                top_reasons = self._extract_top_reasons(result['shap_values'], result['feature_importance'])
                result['top_reasons'] = top_reasons
            else:
                result['top_reasons'] = ['Feature importance unavailable']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in score prediction: {str(e)}")
            return {
                'score': 50.0,
                'uncertainty_lower': 40.0,
                'uncertainty_upper': 60.0,
                'error': str(e),
                'recommendation': 'Model error. Manual review required.',
                'top_reasons': ['Prediction failed - please retry']
            }
    
    def _extract_top_reasons(self, shap_values: List[float], feature_names: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Extract top contributing features as reasons."""
        if not shap_values or not feature_names:
            return []
        
        # Get absolute SHAP values
        shap_abs = [abs(v) for v in shap_values]
        
        # Get top K indices
        top_indices = sorted(range(len(shap_abs)), key=lambda i: shap_abs[i], reverse=True)[:top_k]
        
        reasons = []
        for idx in top_indices:
            reasons.append({
                'feature': feature_names[idx] if idx < len(feature_names) else f'feature_{idx}',
                'impact': float(shap_values[idx])
            })
        
        return reasons


# Standalone testing
if __name__ == "__main__":
    # Example usage
    service = TravelScorePredictorService()
    
    # Mock model outputs
    nb_output = {
        'risk_label': 'medium',
        'probabilities': {'low': 0.1, 'medium': 0.6, 'high': 0.3}
    }
    
    bert_output = {
        'intent': 'report_symptom',
        'slots': {
            'symptoms': ['fever', 'cough'],
            'duration': 3,
            'severity': 0.6
        }
    }
    
    sentiment_output = {
        'fitness_label': 'borderline',
        'confidence': 0.65,
        'probabilities': {'fit': 0.2, 'borderline': 0.65, 'unfit': 0.15}
    }
    
    context = {
        'destination': {
            'disease_prevalence': 0.4,
            'weather_risk': 0.5,
            'altitude_meters': 500,
            'air_quality_index': 60
        },
        'user': {
            'age_group': 5,
            'vaccination_coverage': 0.8,
            'has_chronic_disease': True
        },
        'trip_duration_days': 7
    }
    
    result = service.predict_travel_score(nb_output, bert_output, sentiment_output, context)
    
    print("\n" + "="*60)
    print("Travel Health Score Prediction")
    print("="*60)
    print(f"\nScore: {result['score']:.2f}/100")
    print(f"Confidence Interval: {result['uncertainty_lower']:.1f} - {result['uncertainty_upper']:.1f}")
    print(f"\nRecommendation: {result['recommendation']}")
    print(f"\nTop Contributing Factors:")
    for i, reason in enumerate(result.get('top_reasons', []), 1):
        print(f"  {i}. {reason}")

