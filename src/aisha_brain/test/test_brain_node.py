"""Unit tests for BrainNode — keyword router and LLM fallback."""

import json
import pytest
from unittest.mock import MagicMock, patch
from std_msgs.msg import String

from .conftest import make_string_msg, capture_published


# ---------------------------------------------------------------------------
# _keyword_classify
# ---------------------------------------------------------------------------

class TestKeywordClassify:

    def test_nav_go_to(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('go to the library') == {'intent': 'NAV'}

    def test_nav_navigate_to(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('navigate to the clinic') == {'intent': 'NAV'}

    def test_nav_come_here(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('come here please') == {'intent': 'NAV'}

    def test_nav_take_me_to(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('take me to the cafeteria') == {'intent': 'NAV'}

    def test_action_whatsapp(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('send a whatsapp to my dad') == {'intent': 'ACTION'}

    def test_action_send_message(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('send a message to my mom') == {'intent': 'ACTION'}

    def test_action_remind_me(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('remind me about the meeting') == {'intent': 'ACTION'}

    def test_admin_question_mark(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('What are the school fees?') == {'intent': 'ADMIN'}

    def test_admin_what_prefix(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('what is the school phone number') == {'intent': 'ADMIN'}

    def test_admin_where_prefix(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('where is the swimming pool') == {'intent': 'ADMIN'}

    def test_admin_tell_me(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('tell me about admissions') == {'intent': 'ADMIN'}

    def test_ambiguous_returns_none(self, brain_node):
        node, _ = brain_node
        # Short ambiguous input — no keyword match, not a question
        result = node._keyword_classify('hello')
        assert result is None

    def test_case_insensitive(self, brain_node):
        node, _ = brain_node
        assert node._keyword_classify('GO TO THE POOL') == {'intent': 'NAV'}


# ---------------------------------------------------------------------------
# _llm_classify
# ---------------------------------------------------------------------------

class TestLLMClassify:

    def _mock_llm_response(self, mock_requests, intent):
        mock_requests.post.return_value.json.return_value = {
            'response': json.dumps({'intent': intent})
        }

    def test_llm_admin(self, brain_node):
        node, mock_requests = brain_node
        self._mock_llm_response(mock_requests, 'ADMIN')
        assert node._llm_classify('something ambiguous') == {'intent': 'ADMIN'}

    def test_llm_nav(self, brain_node):
        node, mock_requests = brain_node
        self._mock_llm_response(mock_requests, 'NAV')
        assert node._llm_classify('escort me') == {'intent': 'NAV'}

    def test_llm_action(self, brain_node):
        node, mock_requests = brain_node
        self._mock_llm_response(mock_requests, 'ACTION')
        assert node._llm_classify('ping my teacher') == {'intent': 'ACTION'}

    def test_llm_unknown_intent_falls_back_to_admin(self, brain_node):
        node, mock_requests = brain_node
        mock_requests.post.return_value.json.return_value = {
            'response': json.dumps({'intent': 'UNKNOWN'})
        }
        assert node._llm_classify('xyz') == {'intent': 'ADMIN'}

    def test_llm_connection_error_falls_back_to_admin(self, brain_node):
        import requests as req
        node, mock_requests = brain_node
        mock_requests.post.side_effect = req.ConnectionError
        mock_requests.ConnectionError = req.ConnectionError
        assert node._llm_classify('test') == {'intent': 'ADMIN'}

    def test_llm_timeout_falls_back_to_admin(self, brain_node):
        import requests as req
        node, mock_requests = brain_node
        mock_requests.post.side_effect = req.Timeout
        mock_requests.Timeout = req.Timeout
        assert node._llm_classify('test') == {'intent': 'ADMIN'}

    def test_llm_bad_json_falls_back_to_admin(self, brain_node):
        node, mock_requests = brain_node
        mock_requests.post.return_value.json.return_value = {'response': 'not json {{{'}
        assert node._llm_classify('test') == {'intent': 'ADMIN'}


# ---------------------------------------------------------------------------
# listener_callback routing
# ---------------------------------------------------------------------------

class TestListenerCallback:

    def test_empty_input_ignored(self, brain_node):
        node, _ = brain_node
        admin_pub = capture_published(node, 'admin_pub')
        node.listener_callback(make_string_msg('   '))
        admin_pub.publish.assert_not_called()

    def test_routes_to_admin(self, brain_node):
        node, _ = brain_node
        admin_pub = capture_published(node, 'admin_pub')
        node.listener_callback(make_string_msg('what is the school phone number'))
        admin_pub.publish.assert_called_once()
        payload = json.loads(admin_pub.publish.call_args[0][0].data)
        assert payload['details'] == 'what is the school phone number'

    def test_routes_to_nav(self, brain_node):
        node, _ = brain_node
        nav_pub = capture_published(node, 'nav_pub')
        speech_pub = capture_published(node, 'speech_pub')
        node.listener_callback(make_string_msg('go to the library'))
        nav_pub.publish.assert_called_once()
        # NAV also triggers a TTS "not yet available" message
        speech_pub.publish.assert_called_once()

    def test_routes_to_action(self, brain_node):
        node, _ = brain_node
        action_pub = capture_published(node, 'action_pub')
        node.listener_callback(make_string_msg('send a whatsapp'))
        action_pub.publish.assert_called_once()
