import requests

def test_api():
    API_KEY = "Z0K1F2SYMCHHR8EM"
    
    # Test different endpoints
    endpoints = [
        {
            'function': 'CURRENCY_EXCHANGE_RATE',
            'params': {'from_currency': 'EUR', 'to_currency': 'USD'}
        },
        {
            'function': 'GLOBAL_QUOTE', 
            'params': {'symbol': 'EURUSD'}
        }
    ]
    
    base_url = "https://www.alphavantage.co/query"
    
    for endpoint in endpoints:
        params = endpoint['params']
        params['apikey'] = API_KEY
        params['function'] = endpoint['function']
        
        try:
            response = requests.get(base_url, params=params, timeout=10)
            data = response.json()
            print(f"\nTesting {endpoint['function']}:")
            print(f"Status: {response.status_code}")
            print(f"Response keys: {list(data.keys())}")
            
            if "Error Message" in data:
                print(f"❌ Error: {data['Error Message']}")
            elif "Note" in data:
                print(f"⚠️ Note: {data['Note']}")
            elif "Information" in data:
                print(f"ℹ️ Info: {data['Information']}")
            else:
                print("✅ API working!")
                
        except Exception as e:
            print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_api()