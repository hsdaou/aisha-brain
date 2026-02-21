"""Unit tests for TTSNode — text sanitization and model path resolution."""

import os
import pytest
from unittest.mock import patch, MagicMock
from std_msgs.msg import String

from .conftest import make_string_msg


# ---------------------------------------------------------------------------
# _sanitize_text
# ---------------------------------------------------------------------------

class TestSanitizeText:

    def test_strips_shell_special_chars(self, tts_node):
        result = tts_node._sanitize_text('Hello; rm -rf /')
        assert ';' not in result
        assert 'Hello' in result

    def test_strips_backtick(self, tts_node):
        result = tts_node._sanitize_text('Hello `world`')
        assert '`' not in result

    def test_strips_pipe(self, tts_node):
        result = tts_node._sanitize_text('Hello | world')
        assert '|' not in result

    def test_allows_normal_punctuation(self, tts_node):
        text = "Hello! How are you? I'm fine, thanks."
        result = tts_node._sanitize_text(text)
        assert 'Hello' in result
        assert '!' in result
        assert '?' in result
        assert "'" in result

    def test_truncates_at_2000_chars(self, tts_node):
        long_text = 'a' * 3000
        result = tts_node._sanitize_text(long_text)
        assert len(result) <= 2000

    def test_strips_leading_trailing_whitespace(self, tts_node):
        result = tts_node._sanitize_text('   hello   ')
        assert result == 'hello'

    def test_empty_string_returns_empty(self, tts_node):
        assert tts_node._sanitize_text('') == ''

    def test_only_special_chars_returns_empty(self, tts_node):
        result = tts_node._sanitize_text('`|;&${}[]\\')
        assert result == ''


# ---------------------------------------------------------------------------
# _resolve_model_path
# ---------------------------------------------------------------------------

class TestResolveModelPath:

    def test_absolute_existing_path_returned_as_is(self, tts_node, tmp_path):
        model_file = tmp_path / 'test.onnx'
        model_file.touch()
        result = tts_node._resolve_model_path(str(model_file))
        assert result == str(model_file)

    def test_relative_existing_path_resolved(self, tts_node, tmp_path, monkeypatch):
        model_file = tmp_path / 'voice.onnx'
        model_file.touch()
        monkeypatch.chdir(tmp_path)
        result = tts_node._resolve_model_path('voice.onnx')
        assert os.path.isabs(result)

    def test_not_found_returns_param_unchanged(self, tts_node):
        result = tts_node._resolve_model_path('nonexistent-voice.onnx')
        assert result == 'nonexistent-voice.onnx'

    def test_searches_model_paths(self, tts_node, tmp_path):
        model_file = tmp_path / 'en_US-amy-low.onnx'
        model_file.touch()
        with patch.object(type(tts_node), '_MODEL_SEARCH_PATHS',
                          new_callable=lambda: property(lambda self: [str(tmp_path)])):
            result = tts_node._resolve_model_path('en_US-amy-low.onnx')
            assert result == str(model_file)


# ---------------------------------------------------------------------------
# speak — subprocess handling
# ---------------------------------------------------------------------------

class TestSpeak:

    def test_empty_text_after_sanitize_skips_subprocess(self, tts_node):
        with patch('aisha_brain.tts_node.subprocess.Popen') as mock_popen:
            tts_node.speak(make_string_msg('`|;'))
            mock_popen.assert_not_called()

    def test_normal_text_invokes_piper_and_aplay(self, tts_node):
        mock_piper = MagicMock()
        mock_aplay = MagicMock()
        mock_piper.stdout = MagicMock()
        mock_piper.stdin = MagicMock()

        with patch('aisha_brain.tts_node.subprocess.Popen',
                   side_effect=[mock_piper, mock_aplay]):
            tts_node.speak(make_string_msg('Hello school'))

        mock_piper.stdin.write.assert_called_once()
        mock_aplay.wait.assert_called_once()

    def test_file_not_found_does_not_raise(self, tts_node):
        with patch('aisha_brain.tts_node.subprocess.Popen',
                   side_effect=FileNotFoundError('piper not found')):
            tts_node.speak(make_string_msg('Hello'))  # must not raise

    def test_timeout_kills_processes(self, tts_node):
        import subprocess
        mock_piper = MagicMock()
        mock_aplay = MagicMock()
        mock_piper.stdout = MagicMock()
        mock_piper.stdin = MagicMock()
        mock_aplay.wait.side_effect = subprocess.TimeoutExpired(cmd='aplay', timeout=60)

        with patch('aisha_brain.tts_node.subprocess.Popen',
                   side_effect=[mock_piper, mock_aplay]):
            tts_node.speak(make_string_msg('Hello'))  # must not raise

        mock_piper.kill.assert_called()
        mock_aplay.kill.assert_called()
