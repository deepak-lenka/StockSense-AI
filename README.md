# Financial Statement Analysis with GPT-4

A comprehensive financial analysis system that uses GPT-4 to analyze financial statements, predict earnings, and generate trading signals. Based on academic research, this system combines traditional financial analysis with modern AI techniques.

## Key Features

### 1. Financial Data Processing
- Automated data fetching from Yahoo Finance
- Data standardization and anonymization
- Robust validation and error checking
- Support for multiple financial statements (Balance Sheet, Income Statement, Cash Flow)

### 2. GPT-4 Analysis
- Chain-of-Thought (CoT) prompting for detailed analysis
- Trend analysis of key financial metrics
- Ratio analysis and interpretation
- Future earnings prediction with confidence levels

### 3. Trading Strategy
- Signal generation based on predictions
- Portfolio construction and management
- Performance tracking and optimization
- Risk management features

### 4. Narrative Analysis
- BERT-based embedding generation
- Theme extraction from financial narratives
- Sentiment analysis of predictions
- Key insight identification

### 5. Model Evaluation
- Comprehensive performance metrics
- F1-score calculation
- Confusion matrix analysis
- Performance trend tracking

### 6. Real-time Monitoring
- Continuous data updates
- Price change alerts
- Custom alert callbacks
- Logging and status tracking

## Setup

1. Install Dependencies:
```bash
pip install -r requirements.txt
```

2. Configure Environment:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

3. Run the System:
```bash
python main.py
```

## Project Structure

```
financial_analysis/
├── main.py                 # Main entry point
├── requirements.txt        # Project dependencies
├── README.md              # Documentation
├── financial_data/        # Data processing
│   ├── __init__.py
│   └── data_processor.py
├── analysis/              # Analysis components
│   ├── __init__.py
│   └── narrative_analyzer.py
├── trading/              # Trading strategy
│   ├── __init__.py
│   └── strategy.py
├── evaluation/           # Performance evaluation
│   ├── __init__.py
│   └── evaluator.py
└── monitoring/          # Real-time monitoring
    ├── __init__.py
    └── monitor.py
```

## Usage Guide

### 1. Basic Analysis
```python
from financial_data import FinancialDataProcessor
from analysis import NarrativeAnalyzer

# Initialize components
processor = FinancialDataProcessor()
analyzer = NarrativeAnalyzer()

# Analyze a company
data = processor.fetch_financial_data('AAPL')
analysis = analyzer.analyze_narrative(data)
```

### 2. Trading Strategy
```python
from trading import TradingStrategy

strategy = TradingStrategy()
signals = strategy.generate_signals(predictions)
```

### 3. Monitoring
```python
from monitoring import FinancialMonitor

monitor = FinancialMonitor()
monitor.start_monitoring(['AAPL', 'MSFT', 'GOOGL'])
```

## Performance Metrics

The system tracks various performance metrics:
- Prediction accuracy
- F1-score
- Precision and recall
- Trading signal performance
- Portfolio returns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Citation

If you use this system in your research, please cite:
```
@article{financial_analysis_gpt4,
    title={Financial Statement Analysis with Large Language Models},
    year={2025}
}
```
