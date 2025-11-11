import pandas as pd
import numpy as np
import requests
import ta
from datetime import datetime, timedelta
import warnings
import time
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

class AlphaVantageForexGoldAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Forex pairs and gold symbols for Alpha Vantage
        self.forex_pairs = {
            'EURUSD': 'EURUSD',
            'GBPUSD': 'GBPUSD', 
            'USDJPY': 'USDJPY',
            'USDCHF': 'USDCHF',
            'AUDUSD': 'AUDUSD',
            'USDCAD': 'USDCAD',
            'NZDUSD': 'NZDUSD'
        }
        
        self.gold_symbol = 'GOLD'
        self.timeframe = 'DAILY'
        
    def make_api_request(self, function: str, params: Dict) -> Optional[Dict]:
        """Make API request to Alpha Vantage with error handling"""
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
                return None
            if "Note" in data:
                print(f"API Limit: {data['Note']}")
                time.sleep(60)  # Wait 60 seconds if rate limited
                return self.make_api_request(function, params)
                
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None
    
    def fetch_forex_data(self, from_currency: str, to_currency: str) -> Optional[pd.DataFrame]:
        """Fetch forex data from Alpha Vantage"""
        params = {
            'function': 'FX_DAILY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'outputsize': 'full'
        }
        
        data = self.make_api_request('FX_DAILY', params)
        
        if data and "Time Series FX (Daily)" in data:
            df = pd.DataFrame.from_dict(data["Time Series FX (Daily)"], orient='index')
            df = df.astype(float)
            df.columns = ['Open', 'High', 'Low', 'Close']
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            return df
        else:
            print(f"Failed to fetch data for {from_currency}{to_currency}")
            return None
    
    def fetch_gold_data(self) -> Optional[pd.DataFrame]:
        """Fetch gold data from Alpha Vantage"""
        params = {
            'function': 'GOLD',
            'interval': 'daily',
            'outputsize': 'full'
        }
        
        data = self.make_api_request('GOLD', params)
        
        if data and "data" in data:
            df = pd.DataFrame(data["data"])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.rename(columns={'value': 'Close'})
            df = df.sort_index()
            
            # Create OHLC data (Alpha Vantage gold only provides close)
            df['Open'] = df['Close'].shift(1)
            df['High'] = df[['Open', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'Close']].min(axis=1)
            df = df.dropna()
            
            return df
        else:
            print("Failed to fetch gold data")
            return None
    
    def fetch_crypto_data(self, symbol: str = 'BTC') -> Optional[pd.DataFrame]:
        """Fetch cryptocurrency data (optional)"""
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol,
            'market': 'USD'
        }
        
        data = self.make_api_request('DIGITAL_CURRENCY_DAILY', params)
        
        if data and "Time Series (Digital Currency Daily)" in data:
            ts_data = data["Time Series (Digital Currency Daily)"]
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Rename columns for consistency
            df = df.rename(columns={
                '1a. open (USD)': 'Open',
                '2a. high (USD)': 'High', 
                '3a. low (USD)': 'Low',
                '4a. close (USD)': 'Close',
                '5. volume': 'Volume'
            })
            
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        else:
            print(f"Failed to fetch crypto data for {symbol}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df is None or df.empty:
            return df
            
        # Price-based indicators
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12)
        df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26)
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Histogram'] = macd.macd_diff()
        
        # RSI
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Support and Resistance
        df['Resistance_20'] = df['High'].rolling(window=20).max()
        df['Support_20'] = df['Low'].rolling(window=20).min()
        
        # ATR for volatility
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame, asset_name: str) -> Dict:
        """Generate trading signals based on technical analysis"""
        if df is None or len(df) < 50:
            return {"error": "Insufficient Data"}
            
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        score = 0
        confidence = 0
        
        # Trend Analysis (40% weight)
        if current['SMA_20'] > current['SMA_50'] > current['SMA_200']:
            signals.append("Strong Uptrend (All SMAs Aligned)")
            score += 3
            confidence += 0.4
        elif current['SMA_20'] < current['SMA_50'] < current['SMA_200']:
            signals.append("Strong Downtrend (All SMAs Aligned)") 
            score -= 3
            confidence += 0.4
        elif current['SMA_20'] > current['SMA_50']:
            signals.append("Short-term Uptrend")
            score += 1
            confidence += 0.2
        else:
            signals.append("Short-term Downtrend")
            score -= 1
            confidence += 0.2
        
        # MACD Signal (20% weight)
        if current['MACD'] > current['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            signals.append("MACD Bullish Crossover")
            score += 2
            confidence += 0.2
        elif current['MACD'] < current['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            signals.append("MACD Bearish Crossover")
            score -= 2
            confidence += 0.2
        
        # RSI Analysis (15% weight)
        if current['RSI'] < 30:
            signals.append("Oversold (RSI < 30)")
            score += 1
            confidence += 0.15
        elif current['RSI'] > 70:
            signals.append("Overbought (RSI > 70)")
            score -= 1
            confidence += 0.15
        elif 40 < current['RSI'] < 60:
            signals.append("RSI Neutral")
            confidence += 0.1
        
        # Bollinger Bands (15% weight)
        bb_position = (current['Close'] - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
        if bb_position < 0.2:
            signals.append("Near Lower BB - Potential Support")
            score += 1
            confidence += 0.15
        elif bb_position > 0.8:
            signals.append("Near Upper BB - Potential Resistance") 
            score -= 1
            confidence += 0.15
        
        # Stochastic (10% weight)
        if current['Stoch_K'] < 20 and current['Stoch_D'] < 20:
            signals.append("Stochastic Oversold")
            score += 1
        elif current['Stoch_K'] > 80 and current['Stoch_D'] > 80:
            signals.append("Stochastic Overbought")
            score -= 1
        
        # Determine overall bias
        if score >= 4:
            bias = "STRONG BULLISH"
            risk_level = "LOW"
        elif score >= 2:
            bias = "BULLISH" 
            risk_level = "MEDIUM"
        elif score <= -4:
            bias = "STRONG BEARISH"
            risk_level = "LOW"
        elif score <= -2:
            bias = "BEARISH"
            risk_level = "MEDIUM"
        else:
            bias = "NEUTRAL"
            risk_level = "HIGH"
        
        # Calculate position size suggestion (1-5% of portfolio)
        volatility_factor = min(current['ATR'] / current['Close'], 0.05)  # Cap at 5%
        position_size = max(0.01, 0.05 - volatility_factor)  # 1-5% range
        
        return {
            'asset': asset_name,
            'bias': bias,
            'score': score,
            'confidence': min(confidence, 0.95),
            'risk_level': risk_level,
            'signals': signals,
            'current_price': current['Close'],
            'support': current['Support_20'],
            'resistance': current['Resistance_20'],
            'rsi': current['RSI'],
            'position_size': f"{position_size:.1%}",
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_forex_market(self) -> Dict:
        """Analyze all major forex pairs"""
        print("🔍 ANALYZING FOREX MARKET WITH ALPHA VANTAGE")
        print("=" * 70)
        
        analysis_results = {}
        
        for pair_name, pair_code in self.forex_pairs.items():
            print(f"\n📊 Analyzing {pair_name}...")
            
            data = self.fetch_forex_data(pair_code[:3], pair_code[3:])
            if data is not None:
                data = self.calculate_technical_indicators(data)
                signal = self.generate_signals(data, pair_name)
                
                analysis_results[pair_name] = signal
                
                if 'error' not in signal:
                    print(f"   💰 Price: {signal['current_price']:.5f}")
                    print(f"   🎯 Bias: {signal['bias']} (Score: {signal['score']})")
                    print(f"   📈 Confidence: {signal['confidence']:.1%}")
                    print(f"   ⚠️  Risk: {signal['risk_level']}")
                    print(f"   📊 RSI: {signal['rsi']:.1f}")
                    print(f"   💹 Position Size: {signal['position_size']}")
                    print(f"   🎪 Key Signals: {', '.join(signal['signals'][:3])}")
                
                # Rate limiting delay
                time.sleep(12)  # Alpha Vantage free tier: 1 requests/minute
            else:
                print(f"   ❌ Failed to analyze {pair_name}")
        
        return analysis_results
    
    def analyze_gold(self) -> Dict:
        """Analyze gold market"""
        print(f"\n🥇 ANALYZING GOLD")
        print("=" * 70)
        
        data = self.fetch_gold_data()
        if data is not None:
            data = self.calculate_technical_indicators(data)
            signal = self.generate_signals(data, "GOLD")
            
            if 'error' not in signal:
                print(f"   💰 Price: ${signal['current_price']:.2f}")
                print(f"   🎯 Bias: {signal['bias']} (Score: {signal['score']})")
                print(f"   📈 Confidence: {signal['confidence']:.1%}")
                print(f"   ⚠️  Risk: {signal['risk_level']}")
                print(f"   📊 RSI: {signal['rsi']:.1f}")
                print(f"   📍 Support: ${signal['support']:.2f}")
                print(f"   📍 Resistance: ${signal['resistance']:.2f}")
                print(f"   💹 Position Size: {signal['position_size']}")
                print(f"   🎪 Key Signals:")
                for sig in signal['signals'][:4]:
                    print(f"      • {sig}")
            
            return signal
        else:
            print("   ❌ Failed to analyze Gold")
            return {"error": "Data unavailable"}
    
    def generate_market_report(self, forex_analysis: Dict, gold_analysis: Dict):
        """Generate comprehensive market report"""
        print(f"\n📈 COMPREHENSIVE MARKET REPORT")
        print("=" * 70)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data Source: Alpha Vantage API")
        print("=" * 70)
        
        # Forex Summary
        bullish_pairs = []
        bearish_pairs = []
        
        for pair, analysis in forex_analysis.items():
            if 'bias' in analysis:
                if 'BULL' in analysis['bias']:
                    bullish_pairs.append(pair)
                elif 'BEAR' in analysis['bias']:
                    bearish_pairs.append(pair)
        
        print(f"\n🎯 FOREX MARKET SENTIMENT:")
        print(f"   Bullish Pairs ({len(bullish_pairs)}): {', '.join(bullish_pairs)}")
        print(f"   Bearish Pairs ({len(bearish_pairs)}): {', '.join(bearish_pairs)}")
        
        # Gold Analysis
        if 'bias' in gold_analysis:
            print(f"\n🥇 GOLD ANALYSIS:")
            print(f"   Direction: {gold_analysis['bias']}")
            print(f"   Confidence: {gold_analysis['confidence']:.1%}")
            print(f"   Risk Level: {gold_analysis['risk_level']}")
        
        # Trading Recommendations
        print(f"\n💡 TRADING RECOMMENDATIONS:")
        
        # Find strongest signals
        strong_bullish = []
        strong_bearish = []
        
        for pair, analysis in forex_analysis.items():
            if 'score' in analysis:
                if analysis['score'] >= 3:
                    strong_bullish.append(pair)
                elif analysis['score'] <= -3:
                    strong_bearish.append(pair)
        
        if strong_bullish:
            print(f"   ✅ STRONG BUY SIGNALS: {', '.join(strong_bullish)}")
        if strong_bearish:
            print(f"   🚫 STRONG SELL SIGNALS: {', '.join(strong_bearish)}")
        
        # Market Condition
        total_pairs = len(forex_analysis)
        if len(bearish_pairs) / total_pairs > 0.6:
            print(f"   🌍 OVERALL MARKET: RISK-OFF (USD Strength)")
        elif len(bullish_pairs) / total_pairs > 0.6:
            print(f"   🌍 OVERALL MARKET: RISK-ON (USD Weakness)")
        else:
            print(f"   🌍 OVERALL MARKET: MIXED/NEUTRAL")

# Usage Example
if __name__ == "__main__":
    # Initialize with your API key
    API_KEY = "75RNSPWO51EH25NT"
    
    analyzer = AlphaVantageForexGoldAnalyzer(api_key=API_KEY)
    
    print("🚀 Starting Alpha Vantage Market Analysis...")
    print(f"API Key: {API_KEY[:8]}...{API_KEY[-4:]}")
    
    # Analyze markets
    forex_results = analyzer.analyze_forex_market()
    gold_results = analyzer.analyze_gold()
    
    # Generate comprehensive report
    analyzer.generate_market_report(forex_results, gold_results)
    class AdvancedAlphaVantageAnalyzer(AlphaVantageForexGoldAnalyzer):
        def __init__(self, api_key: str):
            super().__init__(api_key)
        
        def get_market_news(self) -> Optional[Dict]:
            """Get market news from Alpha Vantage"""
            params = {
                'function': 'NEWS_SENTIMENT',
                'topics': 'forex,economy',
                'limit': 5
            }
            
            data = self.make_api_request('NEWS_SENTIMENT', params)
            return data
        
        def get_economic_indicators(self) -> Dict:
            """Get key economic indicators"""
            indicators = {}
            
            # Real GDP (US)
            params_gdp = {
                'function': 'REAL_GDP',
                'interval': 'quarterly'
            }
            indicators['GDP'] = self.make_api_request('REAL_GDP', params_gdp)
            
            # Inflation (CPI)
            params_cpi = {
                'function': 'CPI',
                'interval': 'monthly'
            }
            indicators['CPI'] = self.make_api_request('CPI', params_cpi)
            
            return indicators
        
        def analyze_correlation(self, symbol1: str, symbol2: str) -> float:
            """Analyze correlation between two assets"""
            data1 = self.fetch_forex_data(symbol1[:3], symbol1[3:])
            data2 = self.fetch_forex_data(symbol2[:3], symbol2[3:])
            
            if data1 is not None and data2 is not None:
                # Align dates
                common_dates = data1.index.intersection(data2.index)
                returns1 = data1.loc[common_dates, 'Close'].pct_change().dropna()
                returns2 = data2.loc[common_dates, 'Close'].pct_change().dropna()
                
                correlation = returns1.corr(returns2)
                return correlation
            
            return 0.0

# Enhanced usage with additional features
def run_complete_analysis():
    API_KEY = "Z0K1F2SYMCHHR8EM"
    
    advanced_analyzer = AdvancedAlphaVantageAnalyzer(api_key=API_KEY)
    
    # Basic analysis
    forex_results = advanced_analyzer.analyze_forex_market()
    gold_results = advanced_analyzer.analyze_gold()
    
    # Additional features
    print("\n📰 MARKET NEWS SENTIMENT:")
    news = advanced_analyzer.get_market_news()
    if news and 'feed' in news:
        for item in news['feed'][:3]:
            print(f"   📝 {item['title']}")
            print(f"      Sentiment: {item.get('overall_sentiment_label', 'Unknown')}")
    
    # Correlation analysis
    print("\n🔗 CORRELATION ANALYSIS:")
    eur_usd_corr = advanced_analyzer.analyze_correlation('EUR', 'USD')
    print(f"   EUR/USD vs GBP/USD Correlation: {eur_usd_corr:.2f}")

if __name__ == "__main__":
    run_complete_analysis()