"""Shared pytest fixtures for AI-SHA node unit tests.

ROS2 nodes require rclpy to be initialized before they can be constructed.
These fixtures handle init/shutdown and provide a minimal mock publisher
so tests can inspect published messages without a live ROS2 graph.
"""

import json
import pytest
import rclpy
from unittest.mock import MagicMock, patch


@pytest.fixture(scope='session', autouse=True)
def rclpy_session():
    """Initialize rclpy once per test session."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def brain_node():
    """Return a BrainNode with Ollama calls mocked out."""
    with patch('aisha_brain.brain_node.requests') as mock_requests:
        # Make the startup connectivity check succeed silently
        mock_requests.get.return_value.json.return_value = {
            'models': [{'name': 'gemma3:270m'}]
        }
        from aisha_brain.brain_node import BrainNode
        node = BrainNode()
        yield node, mock_requests
        node.destroy_node()


@pytest.fixture
def action_node():
    """Return an ActionNode ready for testing."""
    from aisha_brain.action_node import ActionNode
    node = ActionNode()
    yield node
    node.destroy_node()


@pytest.fixture
def tts_node():
    """Return a TTSNode with subprocess mocked."""
    from aisha_brain.tts_node import TTSNode
    node = TTSNode()
    yield node
    node.destroy_node()


def make_string_msg(text):
    """Helper: create a ROS2 String message."""
    from std_msgs.msg import String
    msg = String()
    msg.data = text
    return msg


def capture_published(node, publisher_attr):
    """Replace a publisher with a mock that records .publish() calls."""
    mock_pub = MagicMock()
    setattr(node, publisher_attr, mock_pub)
    return mock_pub
