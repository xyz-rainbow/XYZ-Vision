import sys
from unittest.mock import MagicMock

# Mock out heavy dependencies that aren't available in test environment
sys.modules['gradio'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['mss'] = MagicMock()
sys.modules['pytube'] = MagicMock()

# Mock out ultralytics completely
mock_ultralytics = MagicMock()
sys.modules['ultralytics'] = mock_ultralytics
