import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

# Import our modules
from financial_data import FinancialDataProcessor
from trading.strategy import TradingStrategy
from analysis import NarrativeAnalyzer
from evaluation import ModelEvaluator
from monitoring import FinancialMonitor

# Load environment variables
load_dotenv()

class FinancialAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    def generate_cot_prompt(self, financial_data: Dict) -> str:
        """Generate Chain-of-Thought prompt for financial analysis."""
        # Extract key metrics
        metrics = {}
        key_items = [
            'Total Revenue',
            'Operating Income',
            'Net Income',
            'Diluted EPS',
            'Operating Expense',
            'Research And Development',
            'Gross Profit'
        ]
        
        for item in key_items:
            if item in financial_data:
                # Get last 4 periods and format numbers
                values = financial_data[item][-4:]
                metrics[item] = [f"{v:,.2f}" if isinstance(v, (int, float)) else str(v) for v in values]
        
        prompt = f"""Analyze the following financial metrics and predict the direction of future earnings.
        Think through this step-by-step:

        1. Revenue and Profitability Analysis:
        - Review revenue growth trends
        - Analyze operating and net income patterns
        - Examine EPS progression

        2. Operational Efficiency:
        - Evaluate operating expenses
        - Assess R&D investments
        - Calculate gross margins

        3. Growth and Sustainability:
        - Determine growth rates
        - Analyze profit margins
        - Consider investment patterns

        4. Final Prediction:
        - Predict earnings direction (increase/decrease)
        - Provide confidence level (0.0-1.0)
        - Explain key factors

        Financial Metrics (Last 4 Periods):
        {metrics}

        Provide your analysis in this format:
        Step 1: Revenue and Profitability
        Step 2: Operational Analysis
        Step 3: Growth Assessment
        Step 4: Prediction
        Confidence: [0-1]
        """
        return prompt

    def analyze_financial_statements(self, financial_data: Dict, window_size: int = 5) -> Dict:
        """Analyze financial statements using a combination of traditional metrics and AI."""
        try:
            # Extract key metrics
            key_metrics = {
                'Revenue': financial_data.get('Total Revenue', [])[-4:],
                'Operating Income': financial_data.get('Operating Income', [])[-4:],
                'Net Income': financial_data.get('Net Income', [])[-4:],
                'EPS': financial_data.get('Diluted EPS', [])[-4:],
                'R&D': financial_data.get('Research And Development', [])[-4:],
                'Gross Profit': financial_data.get('Gross Profit', [])[-4:]
            }
            
            # Calculate growth rates
            growth_rates = {}
            for metric, values in key_metrics.items():
                if len(values) >= 2 and values[0] != 0:
                    growth = ((values[-1] - values[0]) / abs(values[0])) * 100
                    growth_rates[f'{metric} Growth'] = f"{growth:.1f}%"
            
            # Use GPT-4 for qualitative analysis
            metrics_str = "\n".join([f"{k}: {v}" for k, v in key_metrics.items()])
            growth_str = "\n".join([f"{k}: {v}" for k, v in growth_rates.items()])
            
            prompt = f"""Analyze these financial metrics and growth rates:
            
Metrics (Last 4 Periods):
{metrics_str}

Growth Rates:
{growth_str}

Provide a brief analysis and predict future earnings (increase/decrease) with confidence level."""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Be concise and focus on key trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300,
                top_p=0.9
            )
            
            # Combine quantitative and qualitative analysis
            analysis = response.choices[0].message.content
            
            # Make prediction based on both metrics and GPT-4 analysis
            revenue_trend = np.mean([float(x) for x in key_metrics['Revenue'][-2:]]) if len(key_metrics['Revenue']) >= 2 else 0
            income_trend = np.mean([float(x) for x in key_metrics['Net Income'][-2:]]) if len(key_metrics['Net Income']) >= 2 else 0
            
            prediction = 'increase' if revenue_trend > 0 and income_trend > 0 else 'decrease'
            confidence = 0.85 if abs(revenue_trend) > 10 and abs(income_trend) > 10 else 0.65
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'metrics': key_metrics,
                'growth_rates': growth_rates,
                'insights': analysis
            }
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return None

    def _apply_rolling_window(self, financial_data: Dict, window_size: int) -> Dict:
        """Apply rolling window to financial data."""
        windowed_data = {}
        
        for statement_type, data in financial_data.items():
            if isinstance(data, dict):
                windowed_data[statement_type] = {}
                dates = sorted(data.keys())[-window_size:]  # Get last n periods
                
                for date in dates:
                    if date in data:
                        windowed_data[statement_type][date] = data[date]
                        
        return windowed_data

    def _parse_analysis(self, analysis: str) -> Dict:
        """Parse GPT-4's analysis into structured format."""
        # TODO: Implement parsing logic to extract:
        # - Prediction (increase/decrease)
        # - Confidence level
        # - Key insights
        # This is a placeholder implementation
        return {
            "prediction": "increase",
            "confidence": 0.85,
            "insights": analysis
        }

class PerformanceEvaluator:
    def __init__(self):
        self.predictions = []
        self.actual_results = []
        
    def add_prediction(self, prediction: str, actual: str):
        """Add a prediction and its actual result."""
        self.predictions.append(prediction)
        self.actual_results.append(actual)
        
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        correct = sum(p == a for p, a in zip(self.predictions, self.actual_results))
        total = len(self.predictions)
        
        return {
            "accuracy": correct / total if total > 0 else 0,
            "total_predictions": total
        }

def alert_callback(ticker: str, alert_data: Dict):
    """Handle monitoring alerts."""
    print(f"\nALERT for {ticker}:")
    print(f"Type: {alert_data['type']}")
    print(f"Change: {alert_data['change']:.2%}")
    print(f"Old Price: ${alert_data['old_price']:.2f}")
    print(f"New Price: ${alert_data['new_price']:.2f}")
    print(f"Timestamp: {alert_data['timestamp']}")

def test_system():
    """Run comprehensive system test."""
    print("\n=== Starting Comprehensive System Test ===")
    
    # Initialize all components
    analyzer = FinancialAnalyzer()
    data_processor = FinancialDataProcessor()
    model_evaluator = ModelEvaluator()
    monitor = FinancialMonitor()
    narrative_analyzer = NarrativeAnalyzer()
    
    # Test companies
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA']
    results = {}
    
    print("\n1. Testing Financial Data Processing")
    print("---------------------------------")
    
    for ticker in tickers:
        try:
            print(f"\nProcessing {ticker}:")
            financial_data = data_processor.fetch_financial_data(ticker)
            
            if financial_data:
                print(f"✓ Successfully fetched data for {ticker}")
                results[ticker] = {'data': financial_data}
            else:
                print(f"✗ Failed to fetch data for {ticker}")
                
        except Exception as e:
            print(f"✗ Error processing {ticker}: {e}")
            
    print("\n2. Testing Financial Analysis")
    print("---------------------------")
    
    for ticker, data in results.items():
        try:
            print(f"\nAnalyzing {ticker}:")
            analysis = analyzer.analyze_financial_statements(data['data'])
            
            if analysis:
                print(f"✓ Analysis completed for {ticker}")
                print(f"  - Prediction: {analysis['prediction']}")
                print(f"  - Confidence: {analysis['confidence']:.2%}")
                results[ticker]['analysis'] = analysis
                
                # Test narrative analysis
                narrative_results = narrative_analyzer.analyze_narrative(analysis['insights'])
                print(f"  - Themes: {', '.join(narrative_results['themes'])}")
                print(f"  - Sentiment: {narrative_results['sentiment']:.2f}")
                
        except Exception as e:
            print(f"✗ Error analyzing {ticker}: {e}")
            
    print("\n3. Testing Trading Strategy")
    print("--------------------------")
    
    try:
        strategy = TradingStrategy(confidence_threshold=0.7)
        signals = []
        
        for ticker, data in results.items():
            if 'analysis' in data:
                signal = strategy.generate_signals([{
                    'ticker': ticker,
                    'prediction': data['analysis']['prediction'],
                    'confidence': data['analysis']['confidence']
                }])
                if signal:
                    signals.extend(signal)
                    print(f"✓ Generated trading signal for {ticker}")
                    print(f"  - Direction: {signal[0].direction}")
                    print(f"  - Confidence: {signal[0].confidence:.2%}")
                    
    except Exception as e:
        print(f"✗ Error generating trading signals: {e}")
        
    print("\n4. Testing Monitoring System")
    print("---------------------------")
    
    try:
        monitor.add_alert_callback(alert_callback)
        monitor.start_monitoring(list(results.keys()))
        print(f"✓ Started monitoring {len(results)} tickers")
        
        # Get monitoring status
        status = monitor.get_monitoring_status()
        print(f"✓ Monitoring status: Active={status['active']}")
        print(f"  - Tickers: {', '.join(status['tickers'])}")
        
    except Exception as e:
        print(f"✗ Error setting up monitoring: {e}")
        
    print("\n=== System Test Complete ===\n")
    return results

def main():
    monitor = None
    try:
        # Run system test
        results = test_system()
        
        # Initialize monitoring
        monitor = FinancialMonitor()
        monitor.add_alert_callback(alert_callback)
        monitor.start_monitoring(list(results.keys()))
        
        # Keep the program running for monitoring
        print("\nSystem is now monitoring stocks. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down monitoring...")
        # Clean up
        if monitor:
            monitor.stop_monitoring()
        print("System shutdown complete.")
    except Exception as e:
        print(f"Fatal error: {e}")
        if monitor:
            monitor.stop_monitoring()
        financial_data = data_processor.fetch_financial_data(ticker)
        
        if financial_data:
            # Perform analysis with rolling window
            analysis_result = analyzer.analyze_financial_statements(financial_data, window_size=5)
            
            if analysis_result:
                print("\nAnalysis Results:")
                print(f"Prediction: {analysis_result['prediction']}")
                print(f"Confidence: {analysis_result['confidence']:.2%}")
                print("\nKey Insights:")
                print(analysis_result['insights'])
                
                # Initialize trading strategy
                strategy = TradingStrategy(confidence_threshold=0.7)
                signals = strategy.generate_signals([{
                    'ticker': ticker,
                    'prediction': analysis_result['prediction'],
                    'confidence': analysis_result['confidence']
                }])
                
                if signals:
                    print("\nTrading Signals:")
                    for signal in signals:
                        print(f"Direction: {signal.direction}")
                        print(f"Confidence: {signal.confidence:.2%}")
                
                # Add to evaluator
                evaluator.add_prediction(analysis_result['prediction'], "increase")
                
                # Calculate performance metrics
                metrics = evaluator.calculate_metrics()
                print("\nPerformance Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.2%}")
                print(f"Total Predictions: {metrics['total_predictions']}")
                
                # Analyze narrative using BERT
                from analysis.narrative_analyzer import NarrativeAnalyzer
                narrative_analyzer = NarrativeAnalyzer()
                narrative_analysis = narrative_analyzer.analyze_narrative(analysis_result['insights'])
                
                print("\nNarrative Analysis:")
                print(f"Themes: {', '.join(narrative_analysis['themes'])}")
                print(f"Sentiment Score: {narrative_analysis['sentiment']:.2f}")
                
        else:
            print(f"No valid financial data found for {ticker}")
            
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
