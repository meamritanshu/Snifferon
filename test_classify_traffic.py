import unittest
from unittest.mock import MagicMock, patch
import sys

# Mocking modules that might cause side effects on import or are missing
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
sys.modules['sklearn.ensemble'] = mock_sklearn
sys.modules['sklearn.preprocessing'] = mock_sklearn
sys.modules['sklearn.cluster'] = mock_sklearn
sys.modules['numpy'] = mock_numpy

# Now import the function to test
# We might need to mock get_active_interface specifically if it's called during import
with patch('app.get_active_interface', return_value='eth0'):
    from app import classify_traffic, flow_data

class TestClassifyTraffic(unittest.TestCase):

    def setUp(self):
        flow_data.clear()

    def test_dns_traffic(self):
        packet_features = {
            'protocol': 'DNS',
            'sport': 12345,
            'dport': 53,
            'payload_size': 100
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '8.8.8.8', ('1.2.3.4', '8.8.8.8'))
        self.assertEqual(cls, "DNS Query/Response")
        self.assertEqual(conf, 0.9)

    def test_web_browsing_http(self):
        packet_features = {
            'protocol': 'HTTP',
            'sport': 12345,
            'dport': 80,
            'payload_size': 500
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '93.184.216.34', ('1.2.3.4', '93.184.216.34'))
        self.assertEqual(cls, "Web Browsing")
        self.assertEqual(conf, 0.85)

    def test_streaming_udp(self):
        packet_features = {
            'protocol': 'UDP',
            'sport': 12345,
            'dport': 5000,
            'payload_size': 1200
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
        self.assertEqual(cls, "Streaming")
        self.assertEqual(conf, 0.8)

    def test_streaming_tcp(self):
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 8000,
            'payload_size': 1500
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
        self.assertEqual(cls, "Streaming")
        self.assertEqual(conf, 0.8)

    def test_port_scan_detection(self):
        flow_key = ('1.2.3.4', '5.6.7.8')
        flow_data[flow_key] = {
            'packets': 10,
            'ports': {21, 22, 23, 25, 80, 443}
        }
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 22,
            'payload_size': 0
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', flow_key)
        self.assertEqual(cls, "Port Scan")
        self.assertEqual(conf, 0.95)

    def test_file_transfer(self):
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 443,
            'payload_size': 6000
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
        self.assertEqual(cls, "File Transfer")
        self.assertEqual(conf, 0.75)

    def test_background_service(self):
        packet_features = {
            'protocol': 'UDP',
            'sport': 5353,
            'dport': 5353,
            'payload_size': 50
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '224.0.0.251', ('1.2.3.4', '224.0.0.251'))
        self.assertEqual(cls, "Background Service")
        self.assertEqual(conf, 0.6)

    def test_normal_traffic(self):
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 12345,
            'payload_size': 100
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
        self.assertEqual(cls, "Normal")
        self.assertEqual(conf, 0.7)

    def test_missing_keys(self):
        # This test is expected to FAIL with KeyError currently
        packet_features = {
            'protocol': 'TCP'
            # 'sport', 'dport', 'payload_size' missing
        }
        try:
            cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
            self.assertEqual(cls, "Normal")
        except KeyError as e:
            self.fail(f"classify_traffic raised KeyError: {e}")

    def test_negative_payload(self):
        # Heuristic check: how should it handle negative payload?
        # Currently it might just pass it through or match Rule 5 if payload_size > 5000 (if it was somehow huge negative?)
        # More likely it just stays "Normal".
        packet_features = {
            'protocol': 'TCP',
            'sport': 12345,
            'dport': 80,
            'payload_size': -100
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
        # Should probably be handled gracefully.
        self.assertIn(cls, ["Normal", "Web Browsing"])

    def test_unknown_protocol(self):
        packet_features = {
            'protocol': 'UNKNOWN',
            'sport': 12345,
            'dport': 12345,
            'payload_size': 100
        }
        cls, conf = classify_traffic(packet_features, '1.2.3.4', '5.6.7.8', ('1.2.3.4', '5.6.7.8'))
        self.assertEqual(cls, "Normal")
        self.assertEqual(conf, 0.7)

if __name__ == '__main__':
    unittest.main()
