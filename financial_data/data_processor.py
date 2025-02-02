import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from pathlib import Path

class FinancialDataProcessor:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.company_map = {}
        self.next_company_id = 1

    def _anonymize_company(self, ticker: str) -> str:
        """Anonymize company identifier."""
        if ticker not in self.company_map:
            self.company_map[ticker] = f"Company_{self.next_company_id}"
            self.next_company_id += 1
        return self.company_map[ticker]

    def fetch_financial_data(self, ticker: str, period: str = "5y") -> Dict:
        """Fetch and anonymize financial data."""
        try:
            company = yf.Ticker(ticker)
            
            # Get financial statements
            balance_sheet = company.balance_sheet
            income_stmt = company.income_stmt
            cash_flow = company.cash_flow
            
            # Filter invalid data
            if self._validate_data(balance_sheet, income_stmt, cash_flow):
                # Anonymize and standardize the data
                return self._process_financial_statements(
                    balance_sheet,
                    income_stmt,
                    cash_flow,
                    self._anonymize_company(ticker)
                )
            else:
                print(f"Data validation failed for {ticker}")
                return None
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    def _validate_data(self, balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, 
                       cash_flow: pd.DataFrame) -> bool:
        """Validate financial data for anomalies."""
        try:
            # Print available columns for debugging
            print("Available Balance Sheet items:", balance_sheet.index.tolist())
            print("Available Income Statement items:", income_stmt.index.tolist())
            
            # Check for missing key metrics (using more generic field names)
            bs_fields = balance_sheet.index.tolist()
            is_fields = income_stmt.index.tolist()
            
            # Look for asset-related fields
            asset_fields = [f for f in bs_fields if 'asset' in f.lower()]
            if not asset_fields:
                print("No asset fields found")
                return False
            
            # Look for liability-related fields
            liability_fields = [f for f in bs_fields if 'liabilit' in f.lower()]
            if not liability_fields:
                print("No liability fields found")
                return False
            
            # Look for revenue-related fields
            revenue_fields = [f for f in is_fields if 'revenue' in f.lower() or 'sales' in f.lower()]
            if not revenue_fields:
                print("No revenue fields found")
                return False
            
            # Look for income-related fields
            income_fields = [f for f in is_fields if 'income' in f.lower() or 'earnings' in f.lower()]
            if not income_fields:
                print("No income fields found")
                return False
            
            return True
                
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
            
    def _process_financial_statements(
        self,
        balance_sheet: pd.DataFrame,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame,
        company_id: str
    ) -> Dict:
        """Process and standardize financial statements."""
        # Standardize dates and align statements
        common_dates = sorted(
            set(balance_sheet.columns) &
            set(income_stmt.columns) &
            set(cash_flow.columns)
        )
        
        # Calculate key metrics and ratios
        processed_data = {
            "balance_sheet": balance_sheet[common_dates].to_dict(),
            "income_statement": income_stmt[common_dates].to_dict(),
            "cash_flow": cash_flow[common_dates].to_dict(),
            "metrics": self._calculate_financial_metrics(
                balance_sheet[common_dates],
                income_stmt[common_dates],
                cash_flow[common_dates]
            )
        }
        
        return processed_data
    
    def _calculate_financial_metrics(
        self,
        balance_sheet: pd.DataFrame,
        income_stmt: pd.DataFrame,
        cash_flow: pd.DataFrame
    ) -> Dict:
        """Calculate key financial metrics and ratios."""
        metrics = {}
        
        try:
            # Profitability ratios
            if 'Net Income' in income_stmt.index and 'Total Assets' in balance_sheet.index:
                metrics['roa'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Assets']
            
            if 'Net Income' in income_stmt.index and "Total Stockholder Equity" in balance_sheet.index:
                metrics['roe'] = income_stmt.loc['Net Income'] / balance_sheet.loc['Total Stockholder Equity']
            
            # Liquidity ratios
            if 'Current Assets' in balance_sheet.index and 'Current Liabilities' in balance_sheet.index:
                metrics['current_ratio'] = balance_sheet.loc['Current Assets'] / balance_sheet.loc['Current Liabilities']
            
            # Efficiency ratios
            if 'Total Revenue' in income_stmt.index and 'Total Assets' in balance_sheet.index:
                metrics['asset_turnover'] = income_stmt.loc['Total Revenue'] / balance_sheet.loc['Total Assets']
            
        except Exception as e:
            print(f"Error calculating metrics: {e}")
        
        return metrics
    
    def save_data(self, data: Dict, filename: str):
        """Save processed data to file."""
        output_path = self.data_dir / filename
        pd.to_pickle(data, output_path)
        
    def load_data(self, filename: str) -> Optional[Dict]:
        """Load processed data from file."""
        input_path = self.data_dir / filename
        if input_path.exists():
            return pd.read_pickle(input_path)
        return None
