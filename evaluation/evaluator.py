from typing import List, Dict
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
from datetime import datetime

class ModelEvaluator:
    def __init__(self):
        self.predictions = []
        self.actuals = []
        self.timestamps = []
        self.performance_history = []
        
    def add_prediction(self, prediction: str, actual: str, timestamp: datetime = None):
        """Add a prediction and its actual result."""
        self.predictions.append(1 if prediction == 'increase' else 0)
        self.actuals.append(1 if actual == 'increase' else 0)
        self.timestamps.append(timestamp or datetime.now())
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.predictions:
            return {}
            
        # Convert to numpy arrays
        y_pred = np.array(self.predictions)
        y_true = np.array(self.actuals)
        
        # Calculate metrics
        metrics = {
            'accuracy': np.mean(y_pred == y_true),
            'f1_score': f1_score(y_true, y_pred, average='binary'),
            'total_predictions': len(self.predictions)
        }
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
        
        # Store performance history
        self.performance_history.append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        return metrics
        
    def generate_performance_report(self) -> str:
        """Generate a detailed performance report."""
        metrics = self.calculate_metrics()
        
        report = """
Model Performance Report
=======================

Overall Metrics:
---------------
Accuracy: {accuracy:.2%}
F1-Score: {f1_score:.2%}
Precision: {precision:.2%}
Recall: {recall:.2%}
Specificity: {specificity:.2%}

Total Predictions: {total_predictions}

Performance Trends:
-----------------
""".format(**metrics)

        # Add trend analysis if we have history
        if len(self.performance_history) > 1:
            df = pd.DataFrame(self.performance_history)
            report += f"Accuracy Trend: {'Improving' if df['accuracy'].is_monotonic_increasing else 'Declining'}\n"
            report += f"F1-Score Trend: {'Improving' if df['f1_score'].is_monotonic_increasing else 'Declining'}\n"
            
        return report
