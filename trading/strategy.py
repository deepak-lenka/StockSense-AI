import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class TradingSignal:
    ticker: str
    direction: str  # 'long' or 'short'
    confidence: float
    timestamp: pd.Timestamp

class TradingStrategy:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.positions = {}  # Current positions
        self.trades = []  # Historical trades
        
    def generate_signals(self, predictions: List[Dict]) -> List[TradingSignal]:
        """Generate trading signals from GPT-4 predictions."""
        signals = []
        
        for pred in predictions:
            if pred['confidence'] >= self.confidence_threshold:
                direction = 'long' if pred['prediction'] == 'increase' else 'short'
                
                signals.append(TradingSignal(
                    ticker=pred['ticker'],
                    direction=direction,
                    confidence=pred['confidence'],
                    timestamp=pd.Timestamp.now()
                ))
                
        return signals
    
    def calculate_portfolio_weights(self, signals: List[TradingSignal]) -> Dict[str, float]:
        """Calculate position weights for the portfolio."""
        # Simple equal-weighted portfolio with confidence adjustment
        total_confidence = sum(signal.confidence for signal in signals)
        
        weights = {}
        for signal in signals:
            weight = signal.confidence / total_confidence
            # Adjust sign based on direction
            weights[signal.ticker] = weight if signal.direction == 'long' else -weight
            
        return weights
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate strategy performance metrics."""
        metrics = {
            'total_return': (1 + returns).prod() - 1,
            'annualized_return': (1 + returns).prod() ** (252/len(returns)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
        }
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        excess_returns = returns - risk_free_rate/252
        metrics['sharpe_ratio'] = np.sqrt(252) * excess_returns.mean() / returns.std()
        
        return metrics
    
    def backtest_strategy(
        self,
        signals: List[TradingSignal],
        price_data: Dict[str, pd.DataFrame],
        initial_capital: float = 1000000
    ) -> Tuple[pd.Series, Dict]:
        """Backtest the trading strategy."""
        weights = self.calculate_portfolio_weights(signals)
        
        # Combine price data for all tickers
        portfolio_returns = pd.Series(0, index=next(iter(price_data.values())).index)
        
        for ticker, weight in weights.items():
            if ticker in price_data:
                returns = price_data[ticker]['Close'].pct_change()
                portfolio_returns += returns * weight
        
        # Calculate cumulative portfolio value
        portfolio_value = (1 + portfolio_returns).cumprod() * initial_capital
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio_returns)
        
        return portfolio_value, metrics
    
    def generate_trade_report(self, portfolio_value: pd.Series, metrics: Dict) -> str:
        """Generate a trading strategy performance report."""
        report = """Trading Strategy Performance Report
        =====================================
        
        Portfolio Performance:
        --------------------
        Total Return: {:.2%}
        Annualized Return: {:.2%}
        Sharpe Ratio: {:.2f}
        Volatility: {:.2%}
        Maximum Drawdown: {:.2%}
        
        Initial Portfolio Value: ${:,.2f}
        Final Portfolio Value: ${:,.2f}
        """.format(
            metrics['total_return'],
            metrics['annualized_return'],
            metrics['sharpe_ratio'],
            metrics['volatility'],
            metrics['max_drawdown'],
            portfolio_value.iloc[0],
            portfolio_value.iloc[-1]
        )
        
        return report
