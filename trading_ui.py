import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configure the page
st.set_page_config(
    page_title="Forex & Gold AI Analyzer",
    page_icon="💰",
    layout="wide"
)

class SimpleTradingAnalyzer:
    def __init__(self):
        # Realistic current prices for major pairs (as of Nov 2024)
        self.current_prices = {
            'EUR/USD': 1.0720,
            'GBP/USD': 1.2350,
            'USD/JPY': 151.80,
            'USD/CHF': 0.9050,
            'AUD/USD': 0.6520,
            'USD/CAD': 1.3720,
            'NZD/USD': 0.5920,
            'GOLD': 1985.50
        }
        
        # Realistic support/resistance levels
        self.levels = {
            'EUR/USD': {'support': 1.0650, 'resistance': 1.0820},
            'GBP/USD': {'support': 1.2250, 'resistance': 1.2480},
            'USD/JPY': {'support': 150.20, 'resistance': 153.50},
            'USD/CHF': {'support': 0.8950, 'resistance': 0.9150},
            'AUD/USD': {'support': 0.6450, 'resistance': 0.6620},
            'USD/CAD': {'support': 1.3650, 'resistance': 1.3850},
            'NZD/USD': {'support': 0.5850, 'resistance': 0.6020},
            'GOLD': {'support': 1950.00, 'resistance': 2020.00}
        }
    
    def generate_realistic_data(self, pair, days=100):
        """Generate realistic price data with trends and volatility"""
        base_price = self.current_prices[pair]
        
        # Different volatility for different pairs
        volatilities = {
            'EUR/USD': 0.004, 'GBP/USD': 0.005, 'USD/JPY': 0.008,
            'USD/CHF': 0.004, 'AUD/USD': 0.006, 'USD/CAD': 0.005,
            'NZD/USD': 0.007, 'GOLD': 0.012
        }
        
        volatility = volatilities.get(pair, 0.005)
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)  # For consistent results
        
        # Add some trend based on pair
        trends = {
            'EUR/USD': -0.0001, 'GBP/USD': -0.0002, 'USD/JPY': 0.0003,
            'USD/CHF': 0.0001, 'AUD/USD': -0.0003, 'USD/CAD': 0.0002,
            'NZD/USD': -0.0004, 'GOLD': 0.0005
        }
        
        trend = trends.get(pair, 0.0001)
        
        prices = [base_price]
        for i in range(1, days):
            # Combine trend + random noise
            change = trend + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLC data
        df = pd.DataFrame({
            'Close': prices,
            'Open': [p * (1 + np.random.normal(0, volatility/3)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices]
        }, index=dates)
        
        # Ensure High >= Open, Close and Low <= Open, Close
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI
        df['RSI'] = self.calculate_rsi(df['Close'])
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:window]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / down if down != 0 else 1
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100. / (1. + rs)
        
        for i in range(window, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            rs = up / down if down != 0 else 1
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def analyze_pair(self, pair):
        """Complete analysis for a currency pair"""
        # Generate realistic data
        df = self.generate_realistic_data(pair)
        df = self.calculate_indicators(df)
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signals = []
        score = 0
        
        # 1. Trend Analysis
        if current['SMA_20'] > current['SMA_50']:
            score += 2
            signals.append("📈 Uptrend (SMA20 > SMA50)")
        else:
            score -= 2
            signals.append("📉 Downtrend (SMA20 < SMA50)")
        
        # 2. RSI Analysis
        if current['RSI'] < 30:
            score += 2
            signals.append("🟢 RSI Oversold (<30)")
        elif current['RSI'] > 70:
            score -= 2
            signals.append("🔴 RSI Overbought (>70)")
        elif 40 < current['RSI'] < 60:
            score += 1
            signals.append("⚪ RSI Neutral")
        
        # 3. MACD Analysis
        if current['MACD'] > current['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
            score += 2
            signals.append("✅ MACD Bullish Crossover")
        elif current['MACD'] < current['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
            score -= 2
            signals.append("❌ MACD Bearish Crossover")
        
        # 4. Price vs Support/Resistance
        levels = self.levels[pair]
        current_price = current['Close']
        support = levels['support']
        resistance = levels['resistance']
        
        position = (current_price - support) / (resistance - support)
        
        if position < 0.3:
            score += 2
            signals.append("🛡️ Near Strong Support")
        elif position > 0.7:
            score -= 2
            signals.append("🚧 Near Strong Resistance")
        
        # 5. Bollinger Bands
        bb_position = (current_price - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
        if bb_position < 0.2:
            score += 1
            signals.append("📏 Near Lower BB")
        elif bb_position > 0.8:
            score -= 1
            signals.append("📏 Near Upper BB")
        
        # Determine final signal
        if score >= 6:
            bias = "STRONG BULLISH"
            emoji = "🚀"
            action = "STRONG BUY"
            risk = "LOW"
        elif score >= 3:
            bias = "BULLISH"
            emoji = "📈"
            action = "BUY"
            risk = "MEDIUM"
        elif score <= -6:
            bias = "STRONG BEARISH"
            emoji = "🔻"
            action = "STRONG SELL"
            risk = "LOW"
        elif score <= -3:
            bias = "BEARISH"
            emoji = "📉"
            action = "SELL"
            risk = "MEDIUM"
        else:
            bias = "NEUTRAL"
            emoji = "➡️"
            action = "HOLD"
            risk = "HIGH"
        
        return {
            'pair': pair,
            'bias': bias,
            'emoji': emoji,
            'action': action,
            'risk': risk,
            'score': score,
            'signals': signals,
            'price': current_price,
            'rsi': current['RSI'],
            'support': support,
            'resistance': resistance,
            'chart_data': df
        }
    
    def create_chart(self, df, pair):
        """Create price chart with indicators"""
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            name='SMA 50', 
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'{pair} Price Chart',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_white',
            height=400
        )
        
        return fig

def main():
    st.title("🤖 AI Forex & Gold Trading Analyzer")
    st.markdown("### Real-time Market Analysis & Trading Signals")
    
    # Initialize analyzer
    analyzer = SimpleTradingAnalyzer()
    
    # Sidebar
    st.sidebar.title("Control Panel")
    
    # Asset selection
    assets = [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
        'AUD/USD', 'USD/CAD', 'NZD/USD', 'GOLD'
    ]
    
    selected_assets = st.sidebar.multiselect(
        "Select Assets to Analyze:",
        assets,
        default=['EUR/USD', 'GBP/USD', 'USD/JPY', 'GOLD']
    )
    
    # Analysis button
    if st.sidebar.button("🚀 Analyze Markets", type="primary"):
        if not selected_assets:
            st.warning("Please select at least one asset to analyze.")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, asset in enumerate(selected_assets):
            progress = (i + 1) / len(selected_assets)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {asset}...")
            
            # Analyze the pair
            result = analyzer.analyze_pair(asset)
            results.append(result)
            
            # Small delay for realistic processing feel
            time.sleep(0.5)
        
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        st.success(f"✅ Analysis complete! Processed {len(results)} assets")
        
        # Market Overview
        st.subheader("📊 Market Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        bullish = sum(1 for r in results if 'BULL' in r['bias'])
        bearish = sum(1 for r in results if 'BEAR' in r['bias'])
        neutral = len(results) - bullish - bearish
        
        with col1:
            st.metric("Total Assets", len(results))
        with col2:
            st.metric("Bullish", bullish, delta=f"+{bullish}")
        with col3:
            st.metric("Bearish", bearish, delta=f"-{bearish}")
        with col4:
            st.metric("Neutral", neutral)
        
        # Trading Signals
        st.subheader("💡 Trading Signals")
        
        # Sort by signal strength
        results.sort(key=lambda x: abs(x['score']), reverse=True)
        
        for result in results:
            # Determine card color
            if "STRONG BULL" in result['bias']:
                border_color = "#10b981"
                bg_color = "#f0fdf4"
            elif "BULL" in result['bias']:
                border_color = "#22c55e" 
                bg_color = "#f0fdf4"
            elif "STRONG BEAR" in result['bias']:
                border_color = "#ef4444"
                bg_color = "#fef2f2"
            elif "BEAR" in result['bias']:
                border_color = "#f97316"
                bg_color = "#fef2f2"
            else:
                border_color = "#6b7280"
                bg_color = "#f9fafb"
            
            # Create card
            st.markdown(f"""
            <div style="
                border-left: 5px solid {border_color};
                background-color: {bg_color};
                padding: 1rem;
                border-radius: 5px;
                margin: 0.5rem 0;
            ">
            """, unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
            
            with col1:
                st.markdown(f"### {result['emoji']} {result['pair']}")
                st.markdown(f"**{result['bias']}**")
                
            with col2:
                st.markdown(f"**Action:** `{result['action']}`")
                st.markdown(f"**Risk:** `{result['risk']}` | **Score:** `{result['score']}`")
                st.markdown(f"**RSI:** `{result['rsi']:.1f}`")
                
            with col3:
                if result['pair'] == 'GOLD':
                    st.metric("Price", f"${result['price']:.2f}")
                else:
                    st.metric("Price", f"{result['price']:.5f}")
                    
            with col4:
                st.metric("Support", f"{result['support']:.4f}" if result['pair'] != 'GOLD' else f"${result['support']:.0f}")
                st.metric("Resistance", f"{result['resistance']:.4f}" if result['pair'] != 'GOLD' else f"${result['resistance']:.0f}")
            
            # Expandable details
            with st.expander("View Analysis Details"):
                st.write("**Technical Signals:**")
                for signal in result['signals']:
                    st.write(f"• {signal}")
                
                # Show chart
                chart = analyzer.create_chart(result['chart_data'].tail(60), result['pair'])
                st.plotly_chart(chart, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Risk Assessment
        st.subheader("⚠️ Risk Assessment")
        
        high_risk = sum(1 for r in results if r['risk'] == "HIGH")
        medium_risk = sum(1 for r in results if r['risk'] == "MEDIUM") 
        low_risk = sum(1 for r in results if r['risk'] == "LOW")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.error(f"High Risk: {high_risk} assets")
        with risk_col2:
            st.warning(f"Medium Risk: {medium_risk} assets")
        with risk_col3:
            st.success(f"Low Risk: {low_risk} assets")
        
        # Trading Recommendations
        st.subheader("🎯 Trading Recommendations")
        
        strong_buys = [r for r in results if r['action'] == "STRONG BUY"]
        strong_sells = [r for r in results if r['action'] == "STRONG SELL"]
        
        if strong_buys:
            st.success("**💪 STRONG BUY OPPORTUNITIES:**")
            for trade in strong_buys:
                st.write(f"• **{trade['pair']}** at {trade['price']:.4f} - Target: {trade['resistance']:.4f}")
        
        if strong_sells:
            st.error("**🚨 STRONG SELL OPPORTUNITIES:**")
            for trade in strong_sells:
                st.write(f"• **{trade['pair']}** at {trade['price']:.4f} - Target: {trade['support']:.4f}")
        
        if not strong_buys and not strong_sells:
            st.info("**📊 MARKET NEUTRAL:** No strong trading signals detected. Consider waiting for better opportunities.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This is a demonstration tool for educational purposes only. 
    Always conduct your own research and consult with financial advisors before making trading decisions.
    """)
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()