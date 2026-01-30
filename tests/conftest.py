"""Pytest configuration - mock heavy dependencies."""
import sys
from unittest.mock import MagicMock

# Mock mediapipe before it's imported
mock_mp = MagicMock()
mock_mp.solutions.pose.Pose.return_value = MagicMock()
mock_mp.solutions.pose.PoseLandmark.NOSE = 0
mock_mp.solutions.pose.PoseLandmark.LEFT_SHOULDER = 11
mock_mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER = 12
sys.modules["mediapipe"] = mock_mp

# Mock ultralytics
mock_yolo = MagicMock()
sys.modules["ultralytics"] = mock_yolo
