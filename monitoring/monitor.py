import time
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import threading
import logging
from pathlib import Path

class FinancialMonitor:
    def __init__(self, data_dir: str = "monitoring_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.monitoring_active = False
        self.alert_callbacks = []
        self.monitoring_thread = None
        
        # Set up logging
        logging.basicConfig(
            filename=self.data_dir / 'monitor.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def start_monitoring(self, 
                        tickers: List[str],
                        update_interval: int = 3600,  # 1 hour
                        alert_threshold: float = 0.1   # 10% change
                       ):
        """Start monitoring financial data."""
        self.monitoring_active = True
        self.tickers = tickers
        self.update_interval = update_interval
        self.alert_threshold = alert_threshold
        
        # Start monitoring in a separate thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logging.info(f"Started monitoring {len(tickers)} tickers")
        
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logging.info("Stopped monitoring")
        
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add a callback function for alerts."""
        self.alert_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_values = {ticker: None for ticker in self.tickers}
        
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                for ticker in self.tickers:
                    # Fetch latest data
                    new_data = self._fetch_latest_data(ticker)
                    
                    if new_data:
                        # Check for significant changes
                        if last_values[ticker] is not None:
                            change = (new_data['price'] - last_values[ticker]['price']) / last_values[ticker]['price']
                            
                            if abs(change) >= self.alert_threshold:
                                self._trigger_alerts(ticker, {
                                    'type': 'price_change',
                                    'change': change,
                                    'old_price': last_values[ticker]['price'],
                                    'new_price': new_data['price'],
                                    'timestamp': current_time
                                })
                                
                        last_values[ticker] = new_data
                        
                        # Log the update
                        logging.info(f"Updated data for {ticker}")
                        
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                
            # Wait for next update
            time.sleep(self.update_interval)
            
    def _fetch_latest_data(self, ticker: str) -> Optional[Dict]:
        """Fetch latest financial data for a ticker."""
        try:
            # Implement actual data fetching here
            # This is a placeholder
            return {
                'price': 0.0,  # Replace with actual price
                'timestamp': datetime.now()
            }
        except Exception as e:
            logging.error(f"Error fetching data for {ticker}: {e}")
            return None
            
    def _trigger_alerts(self, ticker: str, alert_data: Dict):
        """Trigger all registered alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(ticker, alert_data)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
                
    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status."""
        return {
            'active': self.monitoring_active,
            'tickers': self.tickers if self.monitoring_active else [],
            'update_interval': self.update_interval if self.monitoring_active else None,
            'last_update': datetime.now().isoformat()
        }
