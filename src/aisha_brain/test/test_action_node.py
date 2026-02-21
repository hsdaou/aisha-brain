"""Unit tests for ActionNode — phone extraction, message parsing, subprocess handling."""

import pytest
from unittest.mock import MagicMock, patch, call
from std_msgs.msg import String

from .conftest import make_string_msg, capture_published


# ---------------------------------------------------------------------------
# Phone number extraction (UAE format: 971XXXXXXXXX)
# ---------------------------------------------------------------------------

class TestSendWhatsapp:

    def test_valid_uae_number(self, action_node):
        with patch('aisha_brain.action_node.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            action_node.send_whatsapp('send whatsapp to 971501234567 saying hello')
            args = mock_run.call_args[0][0]
            assert '971501234567' in args

    def test_no_phone_number_triggers_say(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        action_node.send_whatsapp('send a message with no number')
        speech_pub.publish.assert_called_once()
        assert 'phone number' in speech_pub.publish.call_args[0][0].data.lower()

    def test_message_extracted_after_saying(self, action_node):
        with patch('aisha_brain.action_node.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            action_node.send_whatsapp('send 971501234567 saying hello world')
            args = mock_run.call_args[0][0]
            assert 'hello world' in args

    def test_message_extracted_after_say(self, action_node):
        with patch('aisha_brain.action_node.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            action_node.send_whatsapp('whatsapp 971509999999 say come home')
            args = mock_run.call_args[0][0]
            assert 'come home' in args

    def test_default_message_when_no_content(self, action_node):
        with patch('aisha_brain.action_node.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            action_node.send_whatsapp('whatsapp 971501234567')
            args = mock_run.call_args[0][0]
            assert 'AI-SHA' in args

    def test_subprocess_success_says_sent(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        with patch('aisha_brain.action_node.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            action_node.send_whatsapp('send 971501234567 saying test')
        speech_pub.publish.assert_called_once()
        assert 'sent' in speech_pub.publish.call_args[0][0].data.lower()

    def test_subprocess_failure_says_error(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        with patch('aisha_brain.action_node.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr='error')
            action_node.send_whatsapp('send 971501234567 saying test')
        speech_pub.publish.assert_called_once()
        msg = speech_pub.publish.call_args[0][0].data.lower()
        assert 'trouble' in msg or 'error' in msg or 'try again' in msg

    def test_timeout_handled(self, action_node):
        import subprocess
        speech_pub = capture_published(action_node, 'speech_pub')
        with patch('aisha_brain.action_node.subprocess.run',
                   side_effect=subprocess.TimeoutExpired(cmd='npx', timeout=30)):
            action_node.send_whatsapp('send 971501234567 saying test')
        speech_pub.publish.assert_called_once()
        assert 'long' in speech_pub.publish.call_args[0][0].data.lower()

    def test_file_not_found_handled(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        with patch('aisha_brain.action_node.subprocess.run',
                   side_effect=FileNotFoundError):
            action_node.send_whatsapp('send 971501234567 saying test')
        speech_pub.publish.assert_called_once()
        assert 'not installed' in speech_pub.publish.call_args[0][0].data.lower()


# ---------------------------------------------------------------------------
# handle_action dispatch
# ---------------------------------------------------------------------------

class TestHandleAction:

    def test_whatsapp_keyword_dispatches(self, action_node):
        with patch.object(action_node, 'send_whatsapp') as mock_send:
            action_node.handle_action(make_string_msg('send a whatsapp to dad'))
            mock_send.assert_called_once()

    def test_message_keyword_dispatches(self, action_node):
        with patch.object(action_node, 'send_whatsapp') as mock_send:
            action_node.handle_action(make_string_msg('message my mom'))
            mock_send.assert_called_once()

    def test_calendar_keyword_says_coming_soon(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        action_node.handle_action(make_string_msg('add to my calendar'))
        speech_pub.publish.assert_called_once()
        assert 'coming soon' in speech_pub.publish.call_args[0][0].data.lower()

    def test_unknown_action_handled(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        action_node.handle_action(make_string_msg('do something weird'))
        speech_pub.publish.assert_called_once()

    def test_empty_command_ignored(self, action_node):
        speech_pub = capture_published(action_node, 'speech_pub')
        action_node.handle_action(make_string_msg('   '))
        speech_pub.publish.assert_not_called()
