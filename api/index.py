from flask import Flask, jsonify, request
import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "🚀 Trading AI Flask backend is running!",
        "timestamp": datetime.datetime.now().isoformat()
    })

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'AAPL')
    # Example dummy signal — replace with your model later
    signal = "BUY" if len(symbol) % 2 == 0 else "SELL"
    return jsonify({
        "symbol": symbol,
        "signal": signal
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
