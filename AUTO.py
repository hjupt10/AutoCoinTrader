import os
import re
import time
import hmac
import hashlib
import requests
import pandas as pd
import numpy as np
import base64
import json
import yfinance as yf
import openai
from sklearn.neighbors import KernelDensity
from dotenv import load_dotenv
import logging
from typing import Tuple, Optional, Dict
from datetime import datetime, timezone, timedelta

# Load API keys and webhook URL from .env file
load_dotenv()
ACCESS_KEY = os.getenv('ACCESS_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Coinone API endpoints
API_URL = 'https://api.coinone.co.kr'

# Minimum order quantities
MIN_ORDER_QUANTITY = {
    'btc': 0.0001,   # BTC 최소 주문 수량
    'eth': 0.001,    # ETH 최소 주문 수량
    'xrp': 30,       # XRP 최소 주문 수량
    'sol': 0.1       # SOL 최소 주문 수량 (실제 값으로 수정 필요)
    # 추가 코인에 대한 최소 주문 수량을 필요에 따라 추가하세요
}

# Profit threshold (3%)
PROFIT_THRESHOLD = 1.03

# Maximum trade amount per coin in KRW
MAX_TRADE_AMOUNT = 45000  # 필요에 따라 조정하세요

# Logging configuration
logging.basicConfig(
    filename='trading_bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TradingBot:
    def __init__(self):
        self.access_key = ACCESS_KEY
        self.secret_key = SECRET_KEY
        self.discord_webhook_url = DISCORD_WEBHOOK_URL
        self.openai_api_key = OPENAI_API_KEY
        self.api_url = API_URL
        self.min_order_quantity = MIN_ORDER_QUANTITY
        self.profit_threshold = PROFIT_THRESHOLD
        self.max_trade_amount = MAX_TRADE_AMOUNT
        self.positions = self.initialize_positions()
        openai.api_key = self.openai_api_key
        self.initial_balance = self.get_balance('krw') or 0.0
        self.trade_history_file = 'trade_history.csv'
        self.analysis_file = 'daily_analysis.json'
        self.load_trade_history()
        self.last_hourly_analysis_time = None  # Initialize hourly analysis tracker
        # 추가적인 트레이딩 전략 속성들
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit
        self.trailing_stop_percentage = 0.03  # 3% trailing stop
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_position_size = 0.1  # 10% of account balance
        self.selected_coins = self.get_available_coins()
        self.discord_messages = []

    @staticmethod
    def get_nonce() -> str:
        return str(int(time.time() * 1000))

    @staticmethod
    def get_payload(payload_dict: dict) -> bytes:
        payload_json = json.dumps(payload_dict)
        payload_encoded = base64.b64encode(payload_json.encode('utf-8'))
        return payload_encoded

    def get_headers(self, payload_encoded: bytes) -> dict:
        signature = hmac.new(self.secret_key.encode('utf-8'), payload_encoded, hashlib.sha512).hexdigest()
        return {
            'Content-Type': 'application/json',
            'X-COINONE-PAYLOAD': payload_encoded.decode('utf-8'),
            'X-COINONE-SIGNATURE': signature
        }

    def log_message(self, message: str, level: str = 'info') -> None:
        """
        Log messages to the console and log file.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        if level == 'info':
            logging.info(f"{timestamp} - {message}")
            print(f"{timestamp} - INFO: {message}")
        elif level == 'error':
            logging.error(f"{timestamp} - {message}")
            print(f"{timestamp} - ERROR: {message}")
        elif level == 'warning':
            logging.warning(f"{timestamp} - {message}")
            print(f"{timestamp} - WARNING: {message}")
        elif level == 'debug':
            logging.debug(f"{timestamp} - {message}")
            print(f"{timestamp} - DEBUG: {message}")

    def send_discord_notification(self, message: str) -> None:
        """
        Send a formatted message to Discord webhook.
        """
        if self.discord_webhook_url:
            try:
                data = {"content": message}
                response = requests.post(self.discord_webhook_url, json=data, timeout=10)
                if response.status_code not in [200, 204]:
                    self.log_message(f"Failed to send message to Discord: {response.status_code} {response.text}", level='error')
            except Exception as e:
                self.log_message(f"Error sending message to Discord: {e}", level='error')

        # 메시지를 로컬 리스트에도 추가
        self.discord_messages.append(message)
        # 최대 100개의 메시지만 유지
        self.discord_messages = self.discord_messages[-100:]

    @staticmethod
    def get_available_coins() -> list:
        """
        Fetch the list of available coins from Coinone.
        """
        available_coins = ['btc', 'eth', 'xrp', 'sol']  # Solana 추가
        return available_coins

    @staticmethod
    def get_yahoo_symbol(coinone_symbol: str) -> Optional[str]:
        """
        Map Coinone's symbol to Yahoo Finance's symbol.
        """
        mapping = {
            'btc': 'BTC-USD',
            'eth': 'ETH-USD',
            'xrp': 'XRP-USD',
            'sol': 'SOL-USD'  # Solana 추가
            # 필요한 코인에 대한 매핑을 추가하세요
        }
        return mapping.get(coinone_symbol.lower())

    @staticmethod
    def get_live_data_yahoo(symbol: str, interval: str = '1m', period: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch live data from Yahoo Finance using yfinance.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(interval=interval, period=period)
            if df.empty:
                logging.warning(f"No data fetched for {symbol} from Yahoo Finance.")
                return None
            df = df.reset_index()
            df = df[['Datetime', 'Close', 'Volume']]
            df = df.rename(columns={'Datetime': 'timestamp', 'Close': 'price', 'Volume': 'volume'})
            return df
        except Exception as e:
            logging.error(f"Error fetching live data from Yahoo Finance for {symbol}: {e}")
            return None

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        gain_avg = gain.rolling(window=period).mean()
        loss_avg = loss.rolling(window=period).mean()
        rs = gain_avg / loss_avg
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_env(prices: pd.Series, period: int = 20, deviation: float = 0.025) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Envelope (Upper and Lower Bands).
        """
        sma = prices.rolling(window=period).mean()
        upper_band = sma * (1 + deviation)
        lower_band = sma * (1 - deviation)
        return upper_band, lower_band

    @staticmethod
    def nadaraya_watson_estimator(prices: pd.Series, bandwidth: float = 0.1) -> pd.Series:
        """
        Calculate Nadaraya-Watson Estimator using Kernel Density.
        """
        try:
            X = np.arange(len(prices))[:, np.newaxis]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
            kde.fit(X, sample_weight=prices.values)
            log_density = kde.score_samples(X)
            nw_estimator = np.exp(log_density)
            return pd.Series(nw_estimator, index=prices.index)
        except Exception as e:
            logging.error(f"Error calculating Nadaraya-Watson estimator: {e}")
            return pd.Series(np.nan, index=prices.index)

    def ai_decision_function(self, features: list, daily_analysis: Optional[Dict] = None) -> str:
        """
        Use OpenAI's GPT-4 to decide Buy/Sell/Hold based on features and daily analysis.
        """
        try:
            # Define the prompt with features
            prompt = f"""
You are an advanced investment AI agent. Based on the following market indicators, provide a recommendation by only responding with one of the following options: Buy, Sell, or Hold.

Indicators:
- Current Price: {features[0]:.2f} USD
- RSI: {features[1]:.2f}
- Upper Envelope: {features[2]:.2f} USD
- Lower Envelope: {features[3]:.2f} USD
- Nadaraya-Watson Estimator: {features[4]:.2f}
- Nadaraya-Watson Moving Average: {features[5]:.2f}

Recommendation (only respond with 'Buy', 'Sell', or 'Hold'):
"""

            # Append daily analysis if available
            if daily_analysis:
                prompt += f"""
Based on today's analysis, here are the insights:

{daily_analysis.get('analysis_summary', 'No analysis available.')}

Use these insights to inform your recommendation.

Recommendation (only respond with 'Buy', 'Sell', or 'Hold'):
"""
            else:
                prompt += "Recommendation (only respond with 'Buy', 'Sell', or 'Hold'):"

            # Call OpenAI's GPT-4 model using ChatCompletion API
            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "You are a sophisticated investment AI agent that makes Buy, Sell, or Hold decisions based on technical indicators and past performance analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0,
                n=1,
                stop=["\n"]
            )

            raw_recommendation = response.choices[0].message['content'].strip()

            # Remove any markdown formatting (e.g., **, `) and unnecessary characters
            raw_recommendation = re.sub(r'[\*\`]', '', raw_recommendation)

            # Use regex to extract Buy, Sell, Hold
            match = re.search(r'\b(Buy|Sell|Hold)\b', raw_recommendation, re.IGNORECASE)
            if match:
                recommendation = match.group(1).capitalize()
            else:
                recommendation = 'Hold'  # Default value

            # Validate the recommendation
            if recommendation not in ['Buy', 'Sell', 'Hold']:
                self.log_message(f"Invalid recommendation from AI: '{raw_recommendation}'. Defaulting to Hold.", level='warning')
                recommendation = 'Hold'

            # Log the recommendation
            self.log_message(f"AI Recommendation: {recommendation}")
            return recommendation
        except Exception as e:
            self.log_message(f"Error in AI decision function: {e}", level='error')
            return 'Hold'

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators and generate trading signals.
        """
        df['RSI'] = self.calculate_rsi(df['price'])
        df['UpperENV'], df['LowerENV'] = self.calculate_env(df['price'])
        df['NW_Estimator'] = self.nadaraya_watson_estimator(df['price'])
        df['NW_MA'] = df['NW_Estimator'].rolling(window=5).mean()

        df.dropna(inplace=True)  # Drop rows with NaN values

        # Generate signal
        latest_features = df[['price', 'RSI', 'UpperENV', 'LowerENV', 'NW_Estimator', 'NW_MA']].iloc[-1].values

        # Load daily analysis if available
        daily_analysis = self.load_daily_analysis()

        decision = self.ai_decision_function(latest_features, daily_analysis=daily_analysis)
        df['Signal'] = decision
        return df

    def get_exchange_rate(self) -> Optional[float]:
        """
        Fetch USD to KRW exchange rate.
        """
        try:
            url = 'https://api.exchangerate-api.com/v4/latest/USD'
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            rate = data['rates'].get('KRW', None)
            if rate is None:
                self.log_message("KRW exchange rate not found in response.", level='error')
            return rate
        except requests.exceptions.RequestException as e:
            self.log_message(f"Error fetching exchange rate: {e}", level='error')
            return None

    def get_balance(self, currency: str) -> Optional[float]:
        """
        Fetch the available balance for a specific cryptocurrency from Coinone.
        """
        url = f'{self.api_url}/v2/account/balance/'
        nonce = self.get_nonce()
        payload_dict = {
            'access_token': self.access_key,
            'nonce': nonce
        }
        payload_encoded = self.get_payload(payload_dict)
        headers = self.get_headers(payload_encoded)

        try:
            response = requests.post(url, headers=headers, data=payload_encoded, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self.log_message(f"Error fetching balance for {currency}: {e}", level='error')
            return None

        if data.get('result') != 'success':
            self.log_message(f"Failed to get balance for {currency}: {data.get('errorMsg', 'Unknown Error')}", level='error')
            return None

        try:
            balance_info = data.get(currency)
            if balance_info and 'avail' in balance_info:
                balance = float(balance_info['avail'])
                return balance
            else:
                # If specific currency balance isn't found, log all available balances
                self.log_message(f"No balance information found for {currency}. Available balances:", level='warning')
                for key, value in data.items():
                    if isinstance(value, dict) and 'avail' in value:
                        self.log_message(f"{key.upper()}: {value['avail']}", level='warning')
                return 0.0
        except (KeyError, ValueError, TypeError) as e:
            self.log_message(f"Error parsing balance for {currency}. Response: {data}", level='error')
            return None

    def get_orderbook(self, symbol: str) -> Optional[dict]:
        """
        Fetch the orderbook for a specific cryptocurrency from Coinone.
        """
        url = f'{self.api_url}/orderbook/'
        params = {'currency': symbol}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('result') != 'success':
                self.log_message(f"Failed to get orderbook for {symbol}: {data.get('errorMsg', 'Unknown Error')}", level='error')
                return None
            return data
        except requests.exceptions.RequestException as e:
            self.log_message(f"An error occurred while fetching orderbook for {symbol}: {e}", level='error')
            return None

    @staticmethod
    def get_best_prices(orderbook: dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Extract the best bid and ask prices from the orderbook.
        """
        try:
            bids = orderbook['bid']
            asks = orderbook['ask']
            best_bid = float(bids[0]['price'])
            best_ask = float(asks[0]['price'])
            return best_bid, best_ask
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logging.error(f"Error parsing orderbook data: {e}")
            return None, None

    @staticmethod
    def get_price_unit(price: float) -> Optional[float]:
        """
        Determine the price unit based on Coinone's specifications.
        """
        if 0 <= price < 10:
            return 0.01
        elif 10 <= price < 100:
            return 0.1
        elif 100 <= price < 1000:
            return 1
        elif 1000 <= price < 10000:
            return 5
        elif 10000 <= price < 100000:
            return 10
        elif 100000 <= price < 500000:
            return 50
        elif 500000 <= price < 1000000:
            return 100
        elif 1000000 <= price < 2000000:
            return 500
        elif 2000000 <= price:
            return 1000
        else:
            logging.warning(f"No matching price unit found for price: {price}")
            return None

    @staticmethod
    def adjust_price(price: float, price_unit: float) -> Optional[float]:
        """
        Adjust the price to match the required price unit.
        """
        if price_unit is None:
            logging.warning("Cannot adjust price without valid price unit.")
            return None
        adjusted_price = (price // price_unit) * price_unit
        adjusted_price = round(adjusted_price, int(-np.log10(price_unit)))
        return adjusted_price

    def place_order(self, order_type: str, symbol: str, price: Optional[float] = None, qty: Optional[float] = None, market: bool = False) -> dict:
        """
        Place an order (buy/sell) on Coinone.
        """
        # Check if qty meets the minimum order quantity
        min_qty = self.min_order_quantity.get(symbol.lower(), None)
        if min_qty is None:
            self.log_message(f"No minimum order quantity defined for {symbol}. Skipping order.", level='warning')
            return {'result': 'error', 'errorCode': '999', 'errorMsg': 'No minimum order quantity defined'}

        if qty < min_qty:
            self.log_message(f"Order quantity {qty} for {symbol} is below the minimum of {min_qty}. Skipping order.", level='warning')
            return {'result': 'error', 'errorCode': '999', 'errorMsg': 'Order quantity below minimum'}

        if market:
            # Market order endpoint
            url = f'{self.api_url}/v2/order/market_{order_type}/'
            payload_dict = {
                'access_token': self.access_key,
                'nonce': self.get_nonce(),
                'qty': str(qty),
                'currency': symbol.lower()
            }
        else:
            # Limit order
            if price is None:
                self.log_message("Price must be specified for limit orders.", level='error')
                return {'result': 'error', 'errorCode': '999', 'errorMsg': 'Price not specified'}

            price_unit = self.get_price_unit(price)
            if price_unit is None:
                self.log_message("Failed to get price unit. Order not placed.", level='error')
                return {'result': 'error', 'errorCode': '999', 'errorMsg': 'Failed to get price unit'}

            adjusted_price = self.adjust_price(price, price_unit)
            if adjusted_price is None:
                self.log_message("Failed to adjust price. Order not placed.", level='error')
                return {'result': 'error', 'errorCode': '999', 'errorMsg': 'Failed to adjust price'}

            url = f'{self.api_url}/v2/order/limit_{order_type}/'
            payload_dict = {
                'access_token': self.access_key,
                'nonce': self.get_nonce(),
                'price': str(int(adjusted_price)),
                'qty': str(qty),
                'currency': symbol.lower()
            }

        # Log the request URL and payload
        self.log_message(f"Placing {'Market' if market else 'Limit'} Order:")
        self.log_message(f"URL: {url}")
        self.log_message(f"Payload: {payload_dict}")

        payload_encoded = self.get_payload(payload_dict)
        headers = self.get_headers(payload_encoded)

        try:
            response = requests.post(url, headers=headers, data=payload_encoded, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.HTTPError as http_err:
            self.log_message(f"HTTP error occurred while placing {order_type} order for {symbol}: {http_err}", level='error')
            return {'result': 'error', 'errorCode': str(response.status_code), 'errorMsg': str(http_err)}
        except json.JSONDecodeError as json_err:
            self.log_message(f"JSON decoding failed for {order_type} order for {symbol}: {json_err}", level='error')
            return {'result': 'error', 'errorCode': '999', 'errorMsg': 'Invalid JSON response'}
        except Exception as e:
            self.log_message(f"An unexpected error occurred while placing {order_type} order for {symbol}: {e}", level='error')
            return {'result': 'error', 'errorCode': '999', 'errorMsg': str(e)}

        self.log_message(f"Order Response for {symbol}: {result}")
        self.record_trade(symbol, order_type, price, qty, result)

        # Send order result to Discord if successful
        if result.get('result') == 'success':
            action = 'BUY' if order_type.lower() == 'buy' else 'SELL'
            message = f"**{action} Order Executed:** {qty} {symbol.upper()} at {price:.2f} KRW"
            self.send_discord_notification(message)

        return result

    def initialize_positions(self) -> Dict[str, Dict[str, Optional[float]]]:
        symbols = self.get_available_coins()
        positions = {}
        for symbol in symbols:
            balance = self.get_balance(symbol)
            if balance and balance >= self.min_order_quantity[symbol]:
                positions[symbol] = {'status': 'Long', 'buy_price': None}  # You might want to store the buy price
            else:
                positions[symbol] = {'status': None, 'buy_price': None}
        return positions

    def load_trade_history(self):
        """
        Load trade history from CSV file if exists.
        """
        if os.path.exists(self.trade_history_file):
            try:
                self.trade_history = pd.read_csv(self.trade_history_file)
                self.log_message("Trade history loaded successfully.", level='info')
            except Exception as e:
                self.log_message(f"Error loading trade history: {e}", level='error')
                self.trade_history = pd.DataFrame(columns=['timestamp', 'symbol', 'order_type', 'price', 'quantity', 'result', 'status'])
        else:
            self.trade_history = pd.DataFrame(columns=['timestamp', 'symbol', 'order_type', 'price', 'quantity', 'result', 'status'])

    def record_trade(self, symbol: str, order_type: str, price: Optional[float], qty: Optional[float], result: dict):
        """
        Record each trade into the trade history.
        """
        timestamp = datetime.now(timezone.utc).isoformat()  # ISO 형식으로 저장 (타임존 정보 포함)
        status = 'Success' if result.get('result') == 'success' else 'Failed'
        trade_record = {
            'timestamp': timestamp,
            'symbol': symbol.upper(),
            'order_type': order_type.capitalize(),
            'price': price,
            'quantity': qty,
            'result': json.dumps(result),
            'status': status
        }
        # append를 concat으로 대체
        trade_df = pd.DataFrame([trade_record])
        self.trade_history = pd.concat([self.trade_history, trade_df], ignore_index=True)
        # Save to CSV
        self.trade_history.to_csv(self.trade_history_file, index=False)

    def load_daily_analysis(self) -> Optional[Dict]:
        """
        Load daily analysis from JSON file if exists.
        """
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')  # 수정된 부분
        if os.path.exists(self.analysis_file):
            try:
                with open(self.analysis_file, 'r') as f:
                    analysis_data = json.load(f)
                return analysis_data.get(today, None)
            except Exception as e:
                self.log_message(f"Error loading daily analysis: {e}", level='error')
                return None
        else:
            return None

    def save_daily_analysis(self, analysis: Dict):
        """
        Save daily analysis to JSON file.
        """
        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')  # 수정된 부분
        if os.path.exists(self.analysis_file):
            try:
                with open(self.analysis_file, 'r') as f:
                    analysis_data = json.load(f)
            except Exception as e:
                self.log_message(f"Error loading existing analysis file: {e}", level='error')
                analysis_data = {}
        else:
            analysis_data = {}

        analysis_data[today] = analysis

        try:
            with open(self.analysis_file, 'w') as f:
                json.dump(analysis_data, f, indent=4)
            self.log_message("Daily analysis saved successfully.", level='info')
        except Exception as e:
            self.log_message(f"Error saving daily analysis: {e}", level='error')

    def select_coins(self, coins):
        self.selected_coins = [coin.lower() for coin in coins]
        self.log_message(f"Selected coins updated: {self.selected_coins}")

    def get_performance_data(self):
        # 실제 성능 데이터를 계산하고 반환
        return {
            'total_profit': self.calculate_total_profit(),
            'total_profit_pct': self.calculate_total_profit_percentage(),
            'win_rate': self.calculate_win_rate(),
            'avg_profit': self.calculate_average_profit(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'max_drawdown': self.calculate_max_drawdown()
        }

    def get_trade_history(self):
        # trade_history를 JSON 직렬화 가능한 형식으로 변환
        return self.trade_history.to_dict(orient='records')

    def get_discord_messages(self):
        return self.discord_messages

    # 성능 계산을 위한 헬퍼 메서드들 (실제 구현 필요)
    def calculate_total_profit(self) -> float:
        """
        Calculate the total profit from all successful trades.
        """
        total_profit = 0.0
        buy_orders = {}
        
        for _, trade in self.trade_history.iterrows():
            if trade['status'] != 'Success':
                continue
            
            symbol = trade['symbol']
            order_type = trade['order_type']
            price = float(trade['price'])
            qty = float(trade['quantity'])
            
            if order_type == 'Buy':
                if symbol not in buy_orders:
                    buy_orders[symbol] = []
                buy_orders[symbol].append({'price': price, 'qty': qty})
            elif order_type == 'Sell':
                if symbol in buy_orders and buy_orders[symbol]:
                    buy_order = buy_orders[symbol].pop(0)  # FIFO matching
                    profit = (price - buy_order['price']) * min(qty, buy_order['qty'])
                    total_profit += profit
                    # Adjust quantities if necessary
                    if qty > buy_order['qty']:
                        remaining_qty = qty - buy_order['qty']
                        if remaining_qty > 0 and buy_orders[symbol]:
                            buy_orders[symbol][0]['qty'] -= remaining_qty
                else:
                    # No matching buy order found
                    self.log_message(f"No matching buy order for sell trade: {trade}", level='warning')
        
        return total_profit


    def calculate_total_profit(self) -> float:
        """
        Calculate the total profit from all successful trades.
        """
        total_profit = 0.0
        buy_orders = {}
        
        for _, trade in self.trade_history.iterrows():
            if trade['status'] != 'Success':
                continue  # 성공적인 거래만 고려
            
            symbol = trade['symbol']
            order_type = trade['order_type']
            price = float(trade['price'])
            qty = float(trade['quantity'])
            
            if order_type == 'Buy':
                if symbol not in buy_orders:
                    buy_orders[symbol] = []
                buy_orders[symbol].append({'price': price, 'qty': qty})
            elif order_type == 'Sell':
                if symbol in buy_orders and buy_orders[symbol]:
                    buy_order = buy_orders[symbol].pop(0)  # FIFO 매칭
                    matched_qty = min(qty, buy_order['qty'])
                    profit = (price - buy_order['price']) * matched_qty
                    total_profit += profit
                    # 남은 수량이 있으면 다음 매수 주문과 매칭
                    if qty > matched_qty and buy_orders[symbol]:
                        remaining_qty = qty - matched_qty
                        buy_orders[symbol][0]['qty'] -= remaining_qty
                else:
                    self.log_message(f"No matching buy order for sell trade: {trade}", level='warning')
        
        return total_profit
    def calculate_total_profit_percentage(self) -> float:
        """
        Calculate the total profit as a percentage of the initial account balance.
        Assumes the initial balance is the starting KRW balance plus total profits.
        """
        total_profit = self.calculate_total_profit()
        initial_balance = self.get_initial_balance()
        
        if initial_balance == 0:
            self.log_message("Initial balance is zero. Cannot calculate profit percentage.", level='error')
            return 0.0
        
        profit_percentage = (total_profit / initial_balance) * 100
        return profit_percentage


    def get_initial_balance(self) -> float:
        """
        Retrieve the initial KRW balance when the bot started.
        Assumes this value is stored or can be fetched.
        """
        # If you have a stored initial balance, retrieve it here.
        # For simplicity, let's assume it's stored in a file or set as an attribute.
        if hasattr(self, 'initial_balance'):
            return self.initial_balance
        else:
            # Fetch current KRW balance as initial balance if not set
            balance = self.get_balance('krw')
            self.initial_balance = balance if balance else 0.0
            return self.initial_balance

    def calculate_win_rate(self) -> float:
        """
        Calculate the win rate as the percentage of profitable trades.
        """
        profitable_trades = 0
        total_trades = 0
        buy_orders = {}
        
        for _, trade in self.trade_history.iterrows():
            if trade['status'] != 'Success':
                continue  # 성공적인 거래만 고려
            
            symbol = trade['symbol']
            order_type = trade['order_type']
            price = float(trade['price'])
            qty = float(trade['quantity'])
            
            if order_type == 'Buy':
                if symbol not in buy_orders:
                    buy_orders[symbol] = []
                buy_orders[symbol].append({'price': price, 'qty': qty})
            elif order_type == 'Sell':
                if symbol in buy_orders and buy_orders[symbol]:
                    buy_order = buy_orders[symbol].pop(0)
                    matched_qty = min(qty, buy_order['qty'])
                    profit = (price - buy_order['price']) * matched_qty
                    if profit > 0:
                        profitable_trades += 1
                    total_trades += 1
                    # 남은 수량이 있으면 다음 매수 주문과 매칭
                    if qty > matched_qty and buy_orders[symbol]:
                        remaining_qty = qty - matched_qty
                        buy_orders[symbol][0]['qty'] -= remaining_qty
                else:
                    self.log_message(f"No matching buy order for sell trade: {trade}", level='warning')
        
        if total_trades == 0:
            return 0.0
        
        win_rate = (profitable_trades / total_trades) * 100
        return win_rate



    def calculate_average_profit(self) -> float:
        """
        Calculate the average profit per trade.
        """
        total_profit = 0.0
        total_trades = 0
        buy_orders = {}
        
        for _, trade in self.trade_history.iterrows():
            status = trade['status']
            if status != 'Success':
                continue  # 성공적인 거래만 고려
            
            symbol = trade['symbol']
            order_type = trade['order_type']
            price = float(trade['price'])
            qty = float(trade['quantity'])
            
            if order_type == 'Buy':
                if symbol not in buy_orders:
                    buy_orders[symbol] = []
                buy_orders[symbol].append({'price': price, 'qty': qty})
            elif order_type == 'Sell':
                if symbol in buy_orders and buy_orders[symbol]:
                    buy_order = buy_orders[symbol].pop(0)
                    matched_qty = min(qty, buy_order['qty'])
                    profit = (price - buy_order['price']) * matched_qty
                    total_profit += profit
                    total_trades += 1
                    # 남은 수량이 있으면 다음 매수 주문과 매칭
                    if qty > matched_qty and buy_orders[symbol]:
                        remaining_qty = qty - matched_qty
                        buy_orders[symbol][0]['qty'] -= remaining_qty
                else:
                    self.log_message(f"No matching buy order for sell trade: {trade}", level='warning')
        
        if total_trades == 0:
            return 0.0
        
        average_profit = total_profit / total_trades
        return average_profit



    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the Sharpe Ratio of the trading strategy.
        
        :param risk_free_rate: The risk-free rate of return (default is 0.0)
        :return: Sharpe Ratio
        """
        returns = []
        buy_orders = {}
        
        for _, trade in self.trade_history.iterrows():
            if trade['status'] != 'Success':
                continue  # 성공적인 거래만 고려
            
            symbol = trade['symbol']
            order_type = trade['order_type']
            price = float(trade['price'])
            qty = float(trade['quantity'])
            
            if order_type == 'Buy':
                if symbol not in buy_orders:
                    buy_orders[symbol] = []
                buy_orders[symbol].append({'price': price, 'qty': qty})
            elif order_type == 'Sell':
                if symbol in buy_orders and buy_orders[symbol]:
                    buy_order = buy_orders[symbol].pop(0)
                    matched_qty = min(qty, buy_order['qty'])
                    profit = (price - buy_order['price']) / buy_order['price']  # 수익률 계산
                    returns.append(profit)
                    # 남은 수량이 있으면 다음 매수 주문과 매칭
                    if qty > matched_qty and buy_orders[symbol]:
                        remaining_qty = qty - matched_qty
                        buy_orders[symbol][0]['qty'] -= remaining_qty
                else:
                    self.log_message(f"No matching buy order for sell trade: {trade}", level='warning')
        
        if len(returns) < 2:
            return 0.0  # 샤프 비율 계산에 필요한 데이터 부족
        
        average_return = np.mean(returns)
        return_std = np.std(returns)
        
        if return_std == 0:
            return 0.0
        
        sharpe_ratio = (average_return - risk_free_rate) / return_std
        return sharpe_ratio


    def calculate_max_drawdown(self) -> float:
        """
        Calculate the Maximum Drawdown of the trading strategy.
        """
        balance = self.get_initial_balance()
        peak = balance
        max_drawdown = 0.0
        
        for _, trade in self.trade_history.iterrows():
            if trade['status'] != 'Success':
                continue  # 성공적인 거래만 고려
            
            symbol = trade['symbol']
            order_type = trade['order_type']
            price = float(trade['price'])
            qty = float(trade['quantity'])
            
            if order_type == 'Buy':
                balance -= price * qty  # 매수 시 잔고 감소
            elif order_type == 'Sell':
                balance += price * qty  # 매도 시 잔고 증가
            
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        max_drawdown_percentage = max_drawdown * 100
        return max_drawdown_percentage


    def analyze_daily_performance(self):
        """
        Analyze the day's performance, identifying successes and mistakes.
        """
        try:
            # Define the time window for today (UTC)
            today = datetime.now(timezone.utc).date()
            start_time = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc)
            end_time = start_time + timedelta(days=1)

            # Filter trades for today
            self.trade_history['timestamp'] = pd.to_datetime(self.trade_history['timestamp'], format='ISO8601')
            
            todays_trades = self.trade_history[
                (self.trade_history['timestamp'] >= start_time) &
                (self.trade_history['timestamp'] < end_time)
            ]

            if todays_trades.empty:
                self.log_message("No trades executed today. Skipping analysis.", level='info')
                return

            # Calculate total profit/loss
            total_profit = 0.0
            success_trades = 0
            failed_trades = 0
            buy_prices = {}
            profitable_trades = []
            unprofitable_trades = []

            for _, trade in todays_trades.iterrows():
                symbol = trade['symbol']
                order_type = trade['order_type']
                price = trade['price']
                qty = trade['quantity']
                status = trade['status']

                if order_type == 'Buy' and status == 'Success':
                    buy_prices[symbol] = price
                    success_trades += 1
                elif order_type == 'Sell' and status == 'Success':
                    if symbol in buy_prices:
                        profit = (price - buy_prices[symbol]) * qty
                        total_profit += profit
                        if profit > 0:
                            profitable_trades.append({
                                'symbol': symbol,
                                'buy_price': buy_prices[symbol],
                                'sell_price': price,
                                'profit': profit
                            })
                        else:
                            unprofitable_trades.append({
                                'symbol': symbol,
                                'buy_price': buy_prices[symbol],
                                'sell_price': price,
                                'profit': profit
                            })
                        del buy_prices[symbol]
                        success_trades += 1
                elif status == 'Failed':
                    failed_trades += 1

            # Summarize analysis
            analysis = {
                'total_trades': len(todays_trades),
                'successful_trades': success_trades,
                'failed_trades': failed_trades,
                'total_profit': total_profit,
                'profitable_trades': profitable_trades,
                'unprofitable_trades': unprofitable_trades
            }

            # Identify common mistakes and successes
            if total_profit > 0:
                analysis['analysis_summary'] = (
                    f"**Daily Performance:**\n"
                    f"Total Trades: {len(todays_trades)}\n"
                    f"Successful Trades: {success_trades}\n"
                    f"Failed Trades: {failed_trades}\n"
                    f"Total Profit: {total_profit:.2f} KRW\n"
                    f"Profitable Trades: {len(profitable_trades)}\n"
                    f"Unprofitable Trades: {len(unprofitable_trades)}\n"
                    f"Strategies based on current indicators were effective."
                )
            else:
                analysis['analysis_summary'] = (
                    f"**Daily Performance:**\n"
                    f"Total Trades: {len(todays_trades)}\n"
                    f"Successful Trades: {success_trades}\n"
                    f"Failed Trades: {failed_trades}\n"
                    f"Total Loss: {total_profit:.2f} KRW\n"
                    f"Profitable Trades: {len(profitable_trades)}\n"
                    f"Unprofitable Trades: {len(unprofitable_trades)}\n"
                    f"Review the trading indicators and AI decision prompts for potential improvements."
                )

            # Save analysis
            self.save_daily_analysis(analysis)

            # Log analysis
            self.log_message(f"Daily Analysis Summary: {analysis['analysis_summary']}", level='info')

        except Exception as e:
            self.log_message(f"Error during daily performance analysis: {e}", level='error')

        def analyze_hourly_performance(self):
            """
            Analyze the last 3 hours' performance and send analysis to Discord.
            """
            try:
                # Define the time window for the last 3 hours (UTC)
                now = datetime.now(timezone.utc)
                three_hours_ago = now - timedelta(hours=3)

                # Filter trades for the last 3 hours
                self.trade_history['timestamp'] = pd.to_datetime(self.trade_history['timestamp']).dt.tz_localize(None)
                self.trade_history['timestamp'] = self.trade_history['timestamp'].dt.tz_localize(timezone.utc)
                
                recent_trades = self.trade_history[
                    (self.trade_history['timestamp'] >= three_hours_ago) &
                    (self.trade_history['timestamp'] < now)
                ]

                if recent_trades.empty:
                    analysis_summary = "**Hourly Analysis:**\nNo trades executed in the last 3 hours."
                    self.send_discord_notification(analysis_summary)
                    self.log_message("No trades executed in the last 3 hours. Skipping hourly analysis.", level='info')
                    return

                # Calculate total profit/loss
                total_profit = 0.0
                success_trades = 0
                failed_trades = 0
                buy_prices = {}
                profitable_trades = []
                unprofitable_trades = []

                for _, trade in recent_trades.iterrows():
                    symbol = trade['symbol']
                    order_type = trade['order_type']
                    price = trade['price']
                    qty = trade['quantity']
                    status = trade['status']

                    if order_type == 'Buy' and status == 'Success':
                        buy_prices[symbol] = price
                        success_trades += 1
                    elif order_type == 'Sell' and status == 'Success':
                        if symbol in buy_prices:
                            profit = (price - buy_prices[symbol]) * qty
                            total_profit += profit
                            if profit > 0:
                                profitable_trades.append({
                                    'symbol': symbol,
                                    'buy_price': buy_prices[symbol],
                                    'sell_price': price,
                                    'profit': profit
                                })
                            else:
                                unprofitable_trades.append({
                                    'symbol': symbol,
                                    'buy_price': buy_prices[symbol],
                                    'sell_price': price,
                                    'profit': profit
                                })
                            del buy_prices[symbol]
                            success_trades += 1
                    elif status == 'Failed':
                        failed_trades += 1

                # Summarize analysis
                analysis = {
                    'total_trades': len(recent_trades),
                    'successful_trades': success_trades,
                    'failed_trades': failed_trades,
                    'total_profit': total_profit,
                    'profitable_trades': profitable_trades,
                    'unprofitable_trades': unprofitable_trades
                }

                # Identify common mistakes and successes
                if total_profit > 0:
                    analysis_summary = (
                        f"**Hourly Performance (Last 3 Hours):**\n"
                        f"Total Trades: {len(recent_trades)}\n"
                        f"Successful Trades: {success_trades}\n"
                        f"Failed Trades: {failed_trades}\n"
                        f"Total Profit: {total_profit:.2f} KRW\n"
                        f"Profitable Trades: {len(profitable_trades)}\n"
                        f"Unprofitable Trades: {len(unprofitable_trades)}\n"
                        f"Strategies based on current indicators were effective."
                    )
                else:
                    analysis_summary = (
                        f"**Hourly Performance (Last 3 Hours):**\n"
                        f"Total Trades: {len(recent_trades)}\n"
                        f"Successful Trades: {success_trades}\n"
                        f"Failed Trades: {failed_trades}\n"
                        f"Total Loss: {total_profit:.2f} KRW\n"
                        f"Profitable Trades: {len(profitable_trades)}\n"
                        f"Unprofitable Trades: {len(unprofitable_trades)}\n"
                        f"Review the trading indicators and AI decision prompts for potential improvements."
                    )

                # Send analysis to Discord
                self.send_discord_notification(analysis_summary)

                # Log analysis
                self.log_message(f"Hourly Analysis Summary: {analysis_summary}", level='info')

            except Exception as e:
                self.log_message(f"Error during hourly performance analysis: {e}", level='error')

    def run_daily_analysis(self):
        """
        Perform daily analysis if a day has passed since the last analysis.
        """
        try:
            now = datetime.now(timezone.utc)
            last_analysis_time = getattr(self, 'last_daily_analysis_time', None)
            if last_analysis_time is None or now.date() > last_analysis_time.date():
                self.analyze_daily_performance()
                self.last_daily_analysis_time = now
        except Exception as e:
            self.log_message(f"Error running daily analysis: {e}", level='error')

    def run_hourly_analysis(self):
        """
        Perform hourly analysis every 3 hours.
        """
        try:
            now = datetime.now(timezone.utc)
            if self.last_hourly_analysis_time is None:
                self.last_hourly_analysis_time = now
                return

            elapsed = now - self.last_hourly_analysis_time
            if elapsed >= timedelta(hours=3):
                self.analyze_hourly_performance()
                self.last_hourly_analysis_time = now
        except Exception as e:
            self.log_message(f"Error running hourly analysis: {e}", level='error')

    def calculate_position_size(self, symbol, entry_price):
        account_balance = self.get_balance('KRW')
        if account_balance is None:
            self.log_message("Failed to retrieve KRW balance for position sizing.", level='error')
            return 0.0
        risk_amount = account_balance * self.risk_per_trade
        stop_loss_price = entry_price * (1 - self.stop_loss_percentage)
        position_size = risk_amount / (entry_price - stop_loss_price)
        
        # Limit position size to max_position_size
        max_position_value = account_balance * self.max_position_size
        if position_size * entry_price > max_position_value:
            position_size = max_position_value / entry_price
        
        return position_size

    def update_trailing_stop(self, symbol, current_price):
        position = self.positions[symbol]
        if position['status'] == 'Long':
            new_stop_loss = current_price * (1 - self.trailing_stop_percentage)
            if new_stop_loss > position.get('trailing_stop', 0):
                position['trailing_stop'] = new_stop_loss

    def check_exit_conditions(self, symbol, current_price):
        position = self.positions[symbol]
        if position['status'] == 'Long':
            entry_price = position['buy_price']
            stop_loss = entry_price * (1 - self.stop_loss_percentage)
            take_profit = entry_price * (1 + self.take_profit_percentage)
            trailing_stop = position.get('trailing_stop', stop_loss)

            if current_price <= trailing_stop or current_price >= take_profit:
                return True
        return False

    def main_loop(self):
        symbols = self.get_available_coins()
        while True:
            try:
                # Run daily and hourly analyses
                self.run_daily_analysis()
                self.run_hourly_analysis()
            
                # Add a sleep to prevent excessive looping
                time.sleep(3600)  # Sleep for 1 hour

                # Get exchange rate
                exchange_rate = self.get_exchange_rate()
                if exchange_rate is None:
                    self.log_message("Failed to retrieve exchange rate. Skipping this iteration.", level='warning')
                    time.sleep(60)
                    continue

                for symbol in symbols:
                    yahoo_symbol = self.get_yahoo_symbol(symbol)
                    if yahoo_symbol is None:
                        self.log_message(f"No Yahoo Finance symbol mapping found for {symbol}. Skipping...", level='warning')
                        continue

                    df = self.get_live_data_yahoo(symbol=yahoo_symbol)
                    if df is None or df.empty:
                        self.log_message(f"Failed to retrieve data for {yahoo_symbol}. Skipping...", level='warning')
                        continue

                    # Calculate indicators and generate signals
                    df = self.generate_signals(df)
                    latest_signal = df['Signal'].iloc[-1]
                    latest_price_usd = df['price'].iloc[-1]
                    latest_price_krw = latest_price_usd * exchange_rate

                    # Send current asset status to Discord
                    asset_status_message = (
                        f"**{symbol.upper()} Status:**\n"
                        f"Latest Signal: {latest_signal}\n"
                        f"Price: {latest_price_usd:.2f} USD ({latest_price_krw:.2f} KRW)"
                    )
                    self.send_discord_notification(asset_status_message)

                    self.log_message(f"{symbol.upper()} - Latest Signal: {latest_signal}, Price: {latest_price_usd:.2f} USD")

                    position = self.positions[symbol]['status']
                    buy_price = self.positions[symbol]['buy_price']

                    # Fetch orderbook to get best prices
                    orderbook = self.get_orderbook(symbol)
                    if orderbook is None:
                        self.log_message(f"Failed to get orderbook for {symbol}. Skipping...", level='warning')
                        continue

                    best_bid, best_ask = self.get_best_prices(orderbook)
                    if best_bid is None or best_ask is None:
                        self.log_message(f"Failed to get best prices for {symbol}. Skipping...", level='warning')
                        continue

                    if position == 'Long':
                        # Check if current price has reached profit threshold
                        if latest_price_usd >= buy_price * self.profit_threshold:
                            self.log_message(f"{symbol.upper()} has reached profit threshold. Preparing to sell all holdings.")
                            # Proceed to place sell order at best bid price
                            coin_balance = self.get_balance(symbol)
                            if coin_balance is None or coin_balance == 0:
                                self.log_message(f"No {symbol.upper()} available to sell.", level='warning')
                            else:
                                # Ensure coin_balance meets minimum order quantity
                                min_qty = self.min_order_quantity.get(symbol.lower(), None)
                                if min_qty is None:
                                    self.log_message(f"No minimum order quantity defined for {symbol}. Skipping sell order.", level='warning')
                                    continue
                                if coin_balance < min_qty:
                                    self.log_message(f"Available balance {coin_balance} {symbol.upper()} is below minimum order quantity {min_qty}. Skipping sell order.", level='warning')
                                    continue

                                qty = round(coin_balance, 8)
                                self.log_message(f"Placing Sell Order for {qty} {symbol.upper()} at {best_bid:.2f} KRW")
                                result = self.place_order('sell', symbol=symbol, price=best_bid, qty=qty, market=False)
                                if result.get('result') == 'success':
                                    self.log_message("Sell order placed successfully due to profit threshold.", level='info')
                                    self.positions[symbol] = {'status': None, 'buy_price': None}
                                else:
                                    self.log_message("Sell order failed.", level='error')
                        else:
                            current_profit = (latest_price_usd / buy_price - 1) * 100
                            self.log_message(f"Holding {symbol.upper()} - Current Profit: {current_profit:.2f}%")
                    else:
                        if latest_signal == 'Buy':
                            # Attempt to buy
                            krw_balance = self.get_balance('krw')
                            if krw_balance is None:
                                self.log_message("Failed to retrieve KRW balance.", level='error')
                                continue

                            if krw_balance < self.max_trade_amount:
                                self.log_message("Insufficient KRW balance to place buy order. Skipping...", level='warning')
                                continue

                            # Proceed to place buy order at best ask price
                            self.log_message(f"Preparing to place Buy Order for {symbol.upper()}")
                            qty = self.max_trade_amount / best_ask  # Quantity based on KRW
                            qty = round(qty, 8)

                            # Ensure qty meets minimum order quantity
                            min_qty = self.min_order_quantity.get(symbol.lower(), None)
                            if min_qty is None:
                                self.log_message(f"No minimum order quantity defined for {symbol}. Skipping buy order.", level='warning')
                                continue
                            if qty < min_qty:
                                self.log_message(f"Calculated buy quantity {qty} {symbol.upper()} is below minimum {min_qty}. Skipping buy order.", level='warning')
                                continue

                            self.log_message(f"Placing Buy Order for {qty} {symbol.upper()} at {best_ask:.2f} KRW")
                            result = self.place_order('buy', symbol=symbol, price=best_ask, qty=qty, market=False)
                            if result.get('result') == 'success':
                                # Assume the order was filled at 'best_ask' price
                                executed_price = best_ask
                                self.log_message("Buy order placed successfully.", level='info')
                                self.positions[symbol] = {'status': 'Long', 'buy_price': executed_price}
                            else:
                                self.log_message("Buy order failed.", level='error')
                        elif latest_signal == 'Sell':
                            # Proceed to sell holdings if any
                            coin_balance = self.get_balance(symbol)
                            if coin_balance is None or coin_balance == 0:
                                self.log_message(f"No {symbol.upper()} available to sell.", level='warning')
                            else:
                                # Ensure coin_balance meets minimum order quantity
                                min_qty = self.min_order_quantity.get(symbol.lower(), None)
                                if min_qty is None:
                                    self.log_message(f"No minimum order quantity defined for {symbol}. Skipping sell order.", level='warning')
                                    continue
                                if coin_balance < min_qty:
                                    self.log_message(f"Available balance {coin_balance} {symbol.upper()} is below minimum order quantity {min_qty}. Skipping sell order.", level='warning')
                                    continue

                                self.log_message(f"Preparing to place Sell Order for {symbol.upper()}")
                                qty = round(coin_balance, 8)
                                self.log_message(f"Placing Sell Order for {qty} {symbol.upper()} at {best_bid:.2f} KRW")
                                result = self.place_order('sell', symbol=symbol, price=best_bid, qty=qty, market=False)
                                if result.get('result') == 'success':
                                    self.log_message("Sell order placed successfully.", level='info')
                                    self.positions[symbol] = {'status': None, 'buy_price': None}
                                else:
                                    self.log_message("Sell order failed.", level='error')
                        else:
                            self.log_message(f"No action taken for {symbol.upper()}. Holding or no signal.", level='info')

                    # Sleep for a short duration to avoid hitting API rate limits
                    time.sleep(5)  # 5초 대기, 필요에 따라 조정하세요

            except Exception as e:
                self.log_message(f"An error occurred: {e}", level='error')
                time.sleep(60)
                continue  # Skip to the next iteration of the loop

    def run(self):
        self.log_message("Starting Trading Bot...", level='info')
        self.main_loop()

    def get_trading_status(self):
        # 실제 거래 상태를 확인하는 로직을 구현해야 합니다.
        # 예를 들어, 최근 1분 이내에 거래가 있었다면 'active'를 반환하고
        # 그렇지 않다면 'inactive'를 반환할 수 있습니다.
        return 'active' if self.recent_trade_occurred() else 'inactive'

    def recent_trade_occurred(self):
        # 최근 거래 여부를 확인하는 로직을 구현합니다.
        # 예: 최근 1분 이내의 거래 기록을 확인
        now = datetime.now(timezone.utc)
        one_minute_ago = now - timedelta(minutes=1)
        recent_trades = self.trade_history[self.trade_history['timestamp'] >= one_minute_ago]
        return len(recent_trades) > 0

if __name__ == '__main__':
    bot = TradingBot()  # type: ignore
    bot.run()

 