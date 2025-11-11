import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Professional Forex AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ProfessionalTradingAI:
    def __init__(self):
        # Current market prices (realistic as of late 2024)
        self.market_data = {
            'EUR/USD': {'price': 1.0725, 'support': 1.0650, 'resistance': 1.0820, 'volatility': 0.004},
            'GBP/USD': {'price': 1.2350, 'support': 1.2250, 'resistance': 1.2480, 'volatility': 0.005},
            'USD/JPY': {'price': 151.80, 'support': 150.20, 'resistance': 153.50, 'volatility': 0.008},
            'USD/CHF': {'price': 0.9050, 'support': 0.8950, 'resistance': 0.9150, 'volatility': 0.004},
            'AUD/USD': {'price': 0.6520, 'support': 0.6450, 'resistance': 0.6620, 'volatility': 0.006},
            'USD/CAD': {'price': 1.3720, 'support': 1.3650, 'resistance': 1.3850, 'volatility': 0.005},
            'NZD/USD': {'price': 0.5920, 'support': 0.5850, 'resistance': 0.6020, 'volatility': 0.007},
            'GOLD': {'price': 1985.50, 'support': 1950.00, 'resistance': 2020.00, 'volatility': 0.012}
        }
        
    def generate_market_scenario(self):
        """Generate current market scenario with trends"""
        scenarios = {
            'EUR/USD': {'trend': 'bearish', 'strength': 0.7},
            'GBP/USD': {'trend': 'bearish', 'strength': 0.6},
            'USD/JPY': {'trend': 'bullish', 'strength': 0.8},
            'USD/CHF': {'trend': 'bullish', 'strength': 0.5},
            'AUD/USD': {'trend': 'bearish', 'strength': 0.7},
            'USD/CAD': {'trend': 'bullish', 'strength': 0.6},
            'NZD/USD': {'trend': 'bearish', 'strength': 0.8},
            'GOLD': {'trend': 'bullish', 'strength': 0.9}
        }
        return scenarios
    
    def calculate_technical_indicators(self, df):
        """Calculate professional technical indicators"""
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
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
        
        # Support/Resistance
        df['Resistance'] = df['High'].rolling(20).max()
        df['Support'] = df['Low'].rolling(20).min()
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 1
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 1
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def generate_price_data(self, pair, days=100):
        """Generate realistic price data with current market trends"""
        base_data = self.market_data[pair]
        scenario = self.generate_market_scenario()[pair]
        
        base_price = base_data['price']
        volatility = base_data['volatility']
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)  # Consistent results
        
        # Apply trend based on scenario
        if scenario['trend'] == 'bullish':
            trend_strength = scenario['strength'] * 0.0003
        else:
            trend_strength = -scenario['strength'] * 0.0003
        
        prices = [base_price]
        for i in range(1, days):
            # Trend + noise
            change = trend_strength + np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLC data
        df = pd.DataFrame({
            'Close': prices,
            'Open': [p * (1 + np.random.normal(0, volatility/3)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, volatility/2))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, volatility/2))) for p in prices]
        }, index=dates)
        
        # Ensure proper OHLC relationships
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        return self.calculate_technical_indicators(df)
    
    def analyze_market(self, pair):
        """Complete market analysis for a trading pair"""
        df = self.generate_price_data(pair)
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signals = []
        score = 0
        
        # 1. Trend Analysis (30%)
        if current['SMA_20'] > current['SMA_50']:
            if current['Close'] > current['SMA_20']:
                score += 3
                signals.append("🚀 Strong Uptrend (Price > SMA20 > SMA50)")
            else:
                score += 2
                signals.append("📈 Moderate Uptrend (SMA20 > SMA50)")
        else:
            if current['Close'] < current['SMA_20']:
                score -= 3
                signals.append("🔻 Strong Downtrend (Price < SMA20 < SMA50)")
            else:
                score -= 2
                signals.append("📉 Moderate Downtrend (SMA20 < SMA50)")
        
        # 2. Momentum Analysis (25%)
        if current['RSI'] < 30:
            score += 2
            signals.append("🟢 RSI Oversold - Potential Reversal")
        elif current['RSI'] > 70:
            score -= 2
            signals.append("🔴 RSI Overbought - Potential Pullback")
        elif 45 < current['RSI'] < 55:
            score += 1
            signals.append("⚪ RSI Neutral - Balanced Market")
        
        # 3. MACD Signals (20%)
        if current['MACD'] > current['MACD_Signal'] and previous['MACD'] <= previous['MACD_Signal']:
            score += 2
            signals.append("✅ MACD Bullish Crossover")
        elif current['MACD'] < current['MACD_Signal'] and previous['MACD'] >= previous['MACD_Signal']:
            score -= 2
            signals.append("❌ MACD Bearish Crossover")
        
        # 4. Support/Resistance (15%)
        support = self.market_data[pair]['support']
        resistance = self.market_data[pair]['resistance']
        current_price = current['Close']
        
        position = (current_price - support) / (resistance - support)
        if position < 0.2:
            score += 2
            signals.append("🛡️ Near Strong Support - Buying Opportunity")
        elif position > 0.8:
            score -= 2
            signals.append("🚧 Near Strong Resistance - Selling Pressure")
        
        # 5. Bollinger Bands (10%)
        bb_position = (current_price - current['BB_Lower']) / (current['BB_Upper'] - current['BB_Lower'])
        if bb_position < 0.1:
            score += 1
            signals.append("📏 At Lower Bollinger Band - Oversold")
        elif bb_position > 0.9:
            score -= 1
            signals.append("📏 At Upper Bollinger Band - Overbought")
        
        # Determine final trading signal
        if score >= 7:
            return self.create_signal(pair, "STRONG BUY", "🚀", "green", "LOW", score, signals, current, df)
        elif score >= 4:
            return self.create_signal(pair, "BUY", "📈", "lightgreen", "MEDIUM", score, signals, current, df)
        elif score <= -7:
            return self.create_signal(pair, "STRONG SELL", "🔻", "red", "LOW", score, signals, current, df)
        elif score <= -4:
            return self.create_signal(pair, "SELL", "📉", "lightcoral", "MEDIUM", score, signals, current, df)
        else:
            return self.create_signal(pair, "HOLD", "➡️", "gray", "HIGH", score, signals, current, df)
    
    def create_signal(self, pair, action, emoji, color, risk, score, signals, current, df):
        """Create standardized signal object"""
        return {
            'pair': pair,
            'action': action,
            'emoji': emoji,
            'color': color,
            'risk': risk,
            'score': score,
            'signals': signals,
            'price': current['Close'],
            'rsi': current['RSI'],
            'support': self.market_data[pair]['support'],
            'resistance': self.market_data[pair]['resistance'],
            'chart_data': df
        }
    
    def create_professional_chart(self, df, pair):
        """Create professional trading chart"""
        fig = go.Figure()
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index[-60:],  # Last 60 days
            open=df['Open'][-60:],
            high=df['High'][-60:],
            low=df['Low'][-60:],
            close=df['Close'][-60:],
            name='Price'
        ))
        
        # Moving averages
        fig.add_trace(go.Scatter(
            x=df.index[-60:], y=df['SMA_20'][-60:],
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df.index[-60:], y=df['SMA_50'][-60:],
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title=f'{pair} - Professional Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=400,
            showlegend=True
        )
        
        return fig

def main():
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .signal-card {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎯 PROFESSIONAL FOREX AI TRADING</div>', unsafe_allow_html=True)
    
    # Initialize AI
    ai = ProfessionalTradingAI()
    
    # Sidebar
    st.sidebar.title("🎛️ TRADING DESK")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 ASSET SELECTION")
    assets = list(ai.market_data.keys())
    selected_assets = st.sidebar.multiselect(
        "Choose Trading Pairs:",
        assets,
        default=['EUR/USD', 'GBP/USD', 'USD/JPY', 'GOLD']
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ ANALYSIS SETTINGS")
    analysis_mode = st.sidebar.selectbox(
        "Trading Strategy:",
        ["Intraday Swing", "Position Trading", "Scalping"]
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    # Main Analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📈 LIVE MARKET ANALYSIS")
    with col2:
        if st.button("🔄 UPDATE ANALYSIS", type="primary"):
            st.rerun()
    
    if not selected_assets:
        st.warning("⚠️ Please select trading pairs from the sidebar.")
        st.info("💡 Recommended: EUR/USD, GBP/USD, USD/JPY, GOLD")
        return
    
    # Analysis Progress
    with st.spinner("🤖 AI Analyzing Markets..."):
        progress_bar = st.progress(0)
        results = []
        
        for i, asset in enumerate(selected_assets):
            progress = (i + 1) / len(selected_assets)
            progress_bar.progress(progress)
            
            # Analyze each asset
            result = ai.analyze_market(asset)
            results.append(result)
            
            time.sleep(0.3)  # Realistic processing time
        
        progress_bar.empty()
    
    # Display Results
    st.success(f"✅ ANALYSIS COMPLETE: {len(results)} assets processed")
    
    # Market Overview
    st.subheader("📊 MARKET OVERVIEW")
    
    cols = st.columns(5)
    metrics = [
        ("TOTAL PAIRS", len(results)),
        ("BULLISH", sum(1 for r in results if 'BUY' in r['action'])),
        ("BEARISH", sum(1 for r in results if 'SELL' in r['action'])),
        ("NEUTRAL", sum(1 for r in results if r['action'] == 'HOLD')),
        ("HIGH CONFIDENCE", sum(1 for r in results if abs(r['score']) >= 6))
    ]
    
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.markdown(f'<div class="metric-card"><h3>{value}</h3><p>{label}</p></div>', unsafe_allow_html=True)
    
    # Trading Signals
    st.subheader("💡 TRADING SIGNALS")
    
    # Sort by signal strength
    results.sort(key=lambda x: abs(x['score']), reverse=True)
    
    for result in results:
        # Create signal card
        st.markdown(f"""
        <div class="signal-card" style="border-left-color: {result['color']};">
        """, unsafe_allow_html=True)
        
        # Header row
        col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
        
        with col1:
            st.markdown(f"### {result['emoji']} {result['pair']}")
            st.markdown(f"**{result['action']}**")
            
        with col2:
            st.markdown(f"**CONFIDENCE SCORE:** `{result['score']}/10`")
            st.markdown(f"**RISK LEVEL:** `{result['risk']}`")
            st.markdown(f"**RSI:** `{result['rsi']:.1f}`")
            
        with col3:
            if result['pair'] == 'GOLD':
                st.metric("CURRENT PRICE", f"${result['price']:.2f}")
            else:
                st.metric("CURRENT PRICE", f"{result['price']:.5f}")
                
        with col4:
            if result['pair'] == 'GOLD':
                st.metric("SUPPORT", f"${result['support']:.0f}")
                st.metric("RESISTANCE", f"${result['resistance']:.0f}")
            else:
                st.metric("SUPPORT", f"{result['support']:.4f}")
                st.metric("RESISTANCE", f"{result['resistance']:.4f}")
        
        # Expandable analysis
        with st.expander("📋 VIEW TECHNICAL ANALYSIS", expanded=False):
            st.write("**KEY SIGNALS:**")
            for signal in result['signals']:
                st.write(f"• {signal}")
            
            # Show professional chart
            chart = ai.create_professional_chart(result['chart_data'], result['pair'])
            st.plotly_chart(chart, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Trading Recommendations
    st.subheader("🎯 EXECUTION RECOMMENDATIONS")
    
    strong_signals = [r for r in results if 'STRONG' in r['action']]
    if strong_signals:
        for signal in strong_signals:
            if 'BUY' in signal['action']:
                st.success(f"""
                **🎯 STRONG BUY: {signal['pair']}**
                - Entry: {signal['price']:.4f if signal['pair'] != 'GOLD' else f'${signal['price']:.2f}'}
                - Target: {signal['resistance']:.4f if signal['pair'] != 'GOLD' else f'${signal['resistance']:.0f}'}
                - Stop Loss: {signal['support']:.4f if signal['pair'] != 'GOLD' else f'${signal['support']:.0f}'}
                - Confidence: {abs(signal['score'])}/10
                """)
            else:
                st.error(f"""
                **🎯 STRONG SELL: {signal['pair']}**
                - Entry: {signal['price']:.4f if signal['pair'] != 'GOLD' else f'${signal['price']:.2f}'}
                - Target: {signal['support']:.4f if signal['pair'] != 'GOLD' else f'${signal['support']:.0f}'}
                - Stop Loss: {signal['resistance']:.4f if signal['pair'] != 'GOLD' else f'${signal['resistance']:.0f}'}
                - Confidence: {abs(signal['score'])}/10
                """)
    else:
        st.info("""
        **📊 MARKET ANALYSIS:**
        No strong trading signals detected. Market conditions suggest:
        - Wait for clearer trends
        - Consider smaller position sizes
        - Monitor key support/resistance levels
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p><strong>⚠️ RISK DISCLAIMER:</strong> This is a demonstration tool for educational purposes. 
    Trading involves substantial risk of loss and is not suitable for every investor.</p>
    <p>Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

if __name__ == "__main__":
    main()