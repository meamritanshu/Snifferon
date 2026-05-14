import unittest
from unittest.mock import MagicMock, patch
import sys

# Mocking modules that might cause side effects during import of app.py
mock_flask = MagicMock()
mock_flask_socketio = MagicMock()
mock_scapy = MagicMock()
mock_pandas = MagicMock()
mock_sklearn = MagicMock()
mock_numpy = MagicMock()

sys.modules['flask'] = mock_flask
sys.modules['flask_socketio'] = mock_flask_socketio
sys.modules['scapy'] = mock_scapy
sys.modules['scapy.all'] = mock_scapy
sys.modules['pandas'] = mock_pandas
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.ensemble'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['sklearn.cluster'] = MagicMock()
sys.modules['numpy'] = mock_numpy

# Now import the function to test
from app import classify_traffic

class TestClassifyTraffic(unittest.TestCase):
    def test_dns_traffic(self):
        packet_features = {
            'protocol': 'DNS',
            'sport': 53,
            'dport': 12345,
            'payload_size': 100
        }
        flow_key = ('1.2.3.4', '8.8.8.8')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "DNS Query/Response")
        self.assertEqual(confidence, 0.9)

    def test_web_traffic_http(self):
        packet_features = {
            'protocol': 'HTTP',
            'sport': 80,
            'dport': 54321,
            'payload_size': 500
        }
        flow_key = ('1.2.3.4', '93.184.216.34')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "Web Browsing")
        self.assertEqual(confidence, 0.85)

    def test_web_traffic_https(self):
        packet_features = {
            'protocol': 'HTTPS',
            'sport': 443,
            'dport': 54321,
            'payload_size': 500
        }
        flow_key = ('1.2.3.4', '93.184.216.34')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "Web Browsing")
        self.assertEqual(confidence, 0.85)

    def test_streaming_udp(self):
        packet_features = {
            'protocol': 'UDP',
            'sport': 12345,
            'dport': 5000,
            'payload_size': 1200
        }
        flow_key = ('1.2.3.4', '5.6.7.8')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "Streaming")
        self.assertEqual(confidence, 0.8)

    def test_port_scan(self):
        # Mock flow_data in app.py
        with patch('app.flow_data', {('1.2.3.4', '5.6.7.8'): {'unique_dest_ips': 10, 'packets': 50}}):
            packet_features = {
                'protocol': 'TCP',
                'sport': 12345,
                'dport': 80,
                'payload_size': 0
            }
            flow_key = ('1.2.3.4', '5.6.7.8')
            traffic_class, confidence = classify_traffic(packet_features, flow_key)
            self.assertEqual(traffic_class, "Port Scan")
            self.assertEqual(confidence, 0.95)

    def test_file_transfer(self):
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 80,
            'payload_size': 6000
        }
        flow_key = ('1.2.3.4', '5.6.7.8')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "File Transfer")
        self.assertEqual(confidence, 0.75)

    def test_background_service(self):
        packet_features = {
            'protocol': 'UDP',
            'sport': 5353,
            'dport': 5353,
            'payload_size': 50
        }
        flow_key = ('1.2.3.4', '224.0.0.251')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "Background Service")
        self.assertEqual(confidence, 0.6)

    def test_normal_traffic(self):
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 12345,
            'payload_size': 100
        }
        flow_key = ('1.2.3.4', '5.6.7.8')
        traffic_class, confidence = classify_traffic(packet_features, flow_key)
        self.assertEqual(traffic_class, "Normal")
        self.assertEqual(confidence, 0.7)

if __name__ == '__main__':
    unittest.main()
