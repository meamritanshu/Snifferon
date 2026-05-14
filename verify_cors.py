import sys
from unittest.mock import MagicMock

# Mocking external dependencies
mock_flask = MagicMock()
mock_socketio = MagicMock()
mock_scapy = MagicMock()
mock_pandas = MagicMock()
mock_sklearn = MagicMock()
mock_numpy = MagicMock()

sys.modules['flask'] = mock_flask
sys.modules['flask_socketio'] = mock_socketio
sys.modules['scapy'] = mock_scapy
sys.modules['scapy.all'] = mock_scapy
sys.modules['pandas'] = mock_pandas
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.ensemble'] = mock_sklearn
sys.modules['sklearn.preprocessing'] = mock_sklearn
sys.modules['sklearn.cluster'] = mock_sklearn
sys.modules['numpy'] = mock_numpy

import app

def test_cors_config():
    print("Testing CORS configuration...")
    assert app.DEFAULT_PORT == 5001
    assert f"http://localhost:{app.DEFAULT_PORT}" in app.ALLOWED_ORIGINS
    assert f"http://127.0.0.1:{app.DEFAULT_PORT}" in app.ALLOWED_ORIGINS
    print("CORS config test passed!")

if __name__ == "__main__":
    try:
        test_cors_config()
        print("\nCORS logic verified successfully!")
    except AssertionError as e:
        print("\nCORS logic verification failed!")
        sys.exit(1)
