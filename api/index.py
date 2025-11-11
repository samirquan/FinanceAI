from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "✅ Trading AI Flask backend is running successfully on Vercel!"
    })

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'AAPL')
    signal = "BUY" if len(symbol) % 2 == 0 else "SELL"
    return jsonify({
        "symbol": symbol,
        "signal": signal
    })
