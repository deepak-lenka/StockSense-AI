from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Dict
import pandas as pd

class NarrativeAnalyzer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        
    def generate_embeddings(self, text: str) -> np.ndarray:
        """Generate BERT embeddings for narrative text."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use CLS token embedding as text representation
        embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        return embeddings
        
    def analyze_narrative(self, narrative: str) -> Dict:
        """Analyze narrative for key financial insights."""
        # Generate embeddings
        embeddings = self.generate_embeddings(narrative)
        
        # Extract key themes and sentiment
        themes = self._extract_themes(narrative)
        sentiment = self._analyze_sentiment(narrative)
        
        return {
            'embeddings': embeddings,
            'themes': themes,
            'sentiment': sentiment
        }
        
    def _extract_themes(self, narrative: str) -> List[str]:
        """Extract key financial themes from narrative."""
        key_themes = []
        
        # Financial themes to look for
        themes = {
            'profitability': ['profit', 'margin', 'earnings', 'income'],
            'growth': ['growth', 'increase', 'expand', 'improvement'],
            'risk': ['risk', 'uncertainty', 'volatility', 'exposure'],
            'efficiency': ['efficiency', 'turnover', 'utilization', 'productivity']
        }
        
        narrative = narrative.lower()
        for theme, keywords in themes.items():
            if any(keyword in narrative for keyword in keywords):
                key_themes.append(theme)
                
        return key_themes
        
    def _analyze_sentiment(self, narrative: str) -> float:
        """Analyze sentiment of financial narrative."""
        # Simple rule-based sentiment analysis
        positive_words = ['increase', 'growth', 'improvement', 'positive', 'strong']
        negative_words = ['decrease', 'decline', 'weakness', 'negative', 'poor']
        
        narrative = narrative.lower()
        positive_count = sum(word in narrative for word in positive_words)
        negative_count = sum(word in narrative for word in negative_words)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
            
        return (positive_count - negative_count) / total
