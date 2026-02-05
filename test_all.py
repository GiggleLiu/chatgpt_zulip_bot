"""Tests for the ChatGPT Zulip Bot."""

import pytest
import re
import os
from unittest.mock import patch, MagicMock

import tiktoken

from chatgpt import (
    ChatBot,
    GeneratedFile,
    get_model_specs,
    load_course_materials,
    extract_text_content,
    get_week_context,
    MODEL_SPECS,
)

# =============================================================================
# Model Configuration Tests
# =============================================================================

def test_get_model_specs_exact_match():
    """Test exact model name matching."""
    context, max_output, encoding = get_model_specs("gpt-4o")
    assert context == 128_000
    assert max_output == 16_384
    assert encoding == "o200k_base"


def test_get_model_specs_prefix_match():
    """Test prefix matching for versioned models."""
    context, max_output, encoding = get_model_specs("gpt-4o-2024-12-01")
    assert context == 128_000
    assert encoding == "o200k_base"


def test_get_model_specs_family_fallback():
    """Test fallback based on model family."""
    # GPT-5 family
    context, max_output, _ = get_model_specs("gpt-5.3-preview")
    assert context == 400_000
    
    # GPT-4.1 family
    context, max_output, _ = get_model_specs("gpt-4.1-turbo")
    assert context == 1_047_576
    
    # o-series
    context, max_output, _ = get_model_specs("o3-large")
    assert context == 200_000


def test_get_model_specs_unknown():
    """Test fallback for completely unknown models."""
    context, max_output, encoding = get_model_specs("unknown-model-xyz")
    assert context == 128_000  # default
    assert max_output == 16_384  # default


def test_get_encoding():
    """Test encoding retrieval for different models."""
    _, _, encoding_name = get_model_specs("gpt-4o")
    enc = tiktoken.get_encoding(encoding_name)
    assert enc.name == "o200k_base"
    
    _, _, encoding_name = get_model_specs("gpt-4-turbo")
    enc = tiktoken.get_encoding(encoding_name)
    assert enc.name == "cl100k_base"


def test_all_known_models_have_valid_specs():
    """Verify all models in MODEL_SPECS have valid configurations."""
    for model, (context, max_output, encoding) in MODEL_SPECS.items():
        assert context > 0, f"{model} has invalid context window"
        assert max_output > 0, f"{model} has invalid max output"
        assert encoding in ["cl100k_base", "o200k_base"], f"{model} has invalid encoding"
        assert max_output <= context, f"{model} max_output exceeds context window"


# =============================================================================
# Course Materials Tests
# =============================================================================

def test_load_course_materials():
    """Test course materials loading."""
    materials = load_course_materials()
    assert isinstance(materials, dict)


def test_load_course_materials_missing_dir():
    """Test loading from non-existent directory."""
    materials = load_course_materials("/nonexistent/path")
    assert materials == {}


def test_extract_text_content():
    """Test text extraction from typst content."""
    typst_sample = """#import "@preview/touying:0.6.1": *
#let globalvars = state("t", 0)

= Introduction
== The Simplest Computer

A *finite automaton* is the simplest possible computer.

- States: Red, Green, Yellow
- Alphabet: {tick}
"""
    result = extract_text_content(typst_sample)
    assert "Introduction" in result
    assert "finite automaton" in result
    assert "#import" not in result
    assert "#let" not in result


def test_get_week_context():
    """Test week context generation."""
    mock_materials = {
        "week1": {
            "1.intro.typ": "= Week 1\nIntroduction to DFA"
        },
        "week2": {
            "2.intro.typ": "= Week 2\nNFA and conversions"
        }
    }
    
    # Test specific week
    context = get_week_context(mock_materials, week_num=1)
    assert "WEEK 1" in context
    assert "DFA" in context
    
    # Test non-existent week
    context_empty = get_week_context(mock_materials, week_num=99)
    assert context_empty == ""


def test_get_week_context_max_chars():
    """Test context truncation."""
    mock_materials = {
        "week1": {
            "1.intro.typ": "= Week 1\n" + "A" * 10000
        }
    }
    context = get_week_context(mock_materials, week_num=1, max_chars=100)
    assert len(context) <= 100


def test_get_week_context_with_file_patterns():
    """Test context filtering by file patterns."""
    mock_materials = {
        "week1": {
            "1.learning-sheet.typ": "Learning sheet content",
            "1.validation.typ": "Validation content",
            "1.test.typ": "Test content (should be filtered)",
        }
    }
    
    # With patterns
    context = get_week_context(mock_materials, week_num=1, file_patterns=["*learning-sheet*"])
    assert "Learning sheet" in context
    assert "Validation" not in context
    assert "Test content" not in context


# =============================================================================
# ChatBot Tests
# =============================================================================

def test_chatbot_commands_with_mock():
    """Test bot commands using mock (no API key required)."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key-for-testing")
        
        # Test /reset command - should work even without vector_store_id
        result = bot.get_dm_response("user1", "/reset")
        assert "cleared" in result.lower()
        
        # Test DM without vector store configured - should return error
        assert bot.vector_store_id is None
        result = bot.get_dm_response("user1", "Hello")
        assert "not configured" in result.lower() or "vector" in result.lower()


def test_chatbot_model_detection():
    """Test that model specs are correctly detected."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        # GPT-4o
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        assert bot.max_output_tokens == 16_384
        assert bot.encoding.name == "o200k_base"
        
        # GPT-5.2
        bot = ChatBot(model="gpt-5.2", api_key="fake-key")
        assert bot.max_output_tokens == 32_768
        
        # Custom max_output_tokens override
        bot = ChatBot(model="gpt-4o", api_key="fake-key", max_output_tokens=8000)
        assert bot.max_output_tokens == 8000


def test_chatbot_token_counting():
    """Test token counting via encoding attribute."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        
        # Token counting is done via the encoding attribute
        tokens = len(bot.encoding.encode("Hello, world!"))
        assert tokens > 0
        assert isinstance(tokens, int)


def test_chatbot_agents_sdk_call():
    """Test that the bot calls the Agents SDK correctly for DM mode."""
    with patch('chatgpt.OpenAIClient') as MockClient, \
         patch('chatgpt.Runner') as MockRunner, \
         patch('chatgpt.SQLiteSession') as MockSQLiteSession, \
         patch('chatgpt.OpenAIResponsesCompactionSession') as MockCompactionSession:
        
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Mock session
        mock_session = MagicMock()
        MockCompactionSession.return_value = mock_session
        
        # Mock runner result
        mock_result = MagicMock()
        mock_result.final_output = "This is a test response about DFA."
        mock_result.raw_responses = [MagicMock()]
        mock_result.raw_responses[0].usage.input_tokens = 100
        mock_result.raw_responses[0].usage.output_tokens = 50
        mock_result.raw_responses[0].usage.total_tokens = 150
        MockRunner.run_sync.return_value = mock_result
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key", vector_store_id="vs_123")
        
        result = bot.get_dm_response("user1", "What is a DFA?")
        
        # Verify the response
        assert "test response" in result
        assert "Tokens:" in result
        
        # Verify Runner.run_sync was called
        MockRunner.run_sync.assert_called_once()
        call_kwargs = MockRunner.run_sync.call_args.kwargs
        assert call_kwargs["session"] == mock_session


def test_chatbot_session_reuse():
    """Test that sessions are reused for multi-turn conversations."""
    with patch('chatgpt.OpenAIClient') as MockClient, \
         patch('chatgpt.Runner') as MockRunner, \
         patch('chatgpt.SQLiteSession') as MockSQLiteSession, \
         patch('chatgpt.OpenAIResponsesCompactionSession') as MockCompactionSession:
        
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Mock session
        mock_session = MagicMock()
        MockCompactionSession.return_value = mock_session
        
        # Mock runner result
        mock_result = MagicMock()
        mock_result.final_output = "Response"
        mock_result.raw_responses = [MagicMock()]
        mock_result.raw_responses[0].usage.input_tokens = 100
        mock_result.raw_responses[0].usage.output_tokens = 50
        mock_result.raw_responses[0].usage.total_tokens = 150
        MockRunner.run_sync.return_value = mock_result
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key", vector_store_id="vs_123")
        
        # First message - should create session
        bot.get_dm_response("user1", "Hello")
        assert "user1" in bot.user_sessions
        
        # Second message - should reuse same session
        bot.get_dm_response("user1", "Follow up question")
        
        # Session should only be created once (MockCompactionSession called once per user)
        assert MockCompactionSession.call_count == 1
        
        # But Runner.run_sync should be called twice
        assert MockRunner.run_sync.call_count == 2


def test_chatbot_error_handling():
    """Test error handling when Agents SDK fails."""
    with patch('chatgpt.OpenAIClient') as MockClient, \
         patch('chatgpt.Runner') as MockRunner, \
         patch('chatgpt.SQLiteSession') as MockSQLiteSession, \
         patch('chatgpt.OpenAIResponsesCompactionSession') as MockCompactionSession:
        
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Mock session
        mock_session = MagicMock()
        MockCompactionSession.return_value = mock_session
        
        # Make Runner.run_sync fail
        MockRunner.run_sync.side_effect = Exception("API error")
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key", vector_store_id="vs_123")
        
        result = bot.get_dm_response("user1", "Hello")
        
        # Should return error message, not crash
        assert "Error" in result or "error" in result


# =============================================================================
# Integration Tests (require API key)
# =============================================================================

@pytest.fixture
def chatbot_with_api():
    """Create ChatBot with real API key (skips if not available)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    vector_store_id = os.environ.get("VECTOR_STORE_ID")
    if not vector_store_id:
        pytest.skip("VECTOR_STORE_ID environment variable not set")
    
    model = os.environ.get("MODEL", "gpt-4o")
    return ChatBot(model=model, api_key=api_key, vector_store_id=vector_store_id)


def test_real_api_response(chatbot_with_api):
    """Test actual API response (requires API key)."""
    response = chatbot_with_api.get_dm_response("test_user", "What is a DFA? Answer in one sentence.")
    assert isinstance(response, str)
    assert len(response) > 0
    # Clean up
    chatbot_with_api.clear_user_session("test_user")


# =============================================================================
# Zulip Mention Pattern Tests
# =============================================================================

class TestMentionPatterns:
    """Test Zulip mention pattern matching."""
    
    @pytest.fixture
    def mention_pattern(self):
        """Create mention pattern for bot named 'ChatGPT'."""
        bot_name = "ChatGPT"
        return rf"@_?\*\*{re.escape(bot_name)}(\|\d+)?\*\*"
    
    def test_regular_mention(self, mention_pattern):
        """Test regular @**BotName** mention."""
        message = "@**ChatGPT** What is a DFA?"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "What is a DFA?"
    
    def test_quote_reply_mention(self, mention_pattern):
        """Test quote reply @_**BotName|ID** mention."""
        message = "@_**ChatGPT|132** Can you explain more?"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "Can you explain more?"
    
    def test_mention_with_id_no_underscore(self, mention_pattern):
        """Test @**BotName|ID** mention (without underscore)."""
        message = "@**ChatGPT|132** Another question"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "Another question"
    
    def test_mention_not_at_start(self, mention_pattern):
        """Test that mention not at start doesn't match."""
        message = "Hello @**ChatGPT** how are you?"
        assert not re.match(mention_pattern, message)
    
    def test_mention_with_multiline(self, mention_pattern):
        """Test mention with multiline message."""
        message = "@**ChatGPT** First line\nSecond line\nThird line"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "First line\nSecond line\nThird line"
    
    def test_mention_only(self, mention_pattern):
        """Test message that is only a mention."""
        message = "@**ChatGPT**"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == ""
    
    def test_wrong_bot_name(self, mention_pattern):
        """Test that wrong bot name doesn't match."""
        message = "@**OtherBot** Hello"
        assert not re.match(mention_pattern, message)
    
    def test_quote_reply_with_quote_block(self, mention_pattern):
        """Test stripping quote blocks from quote-replies."""
        message = '''@_**ChatGPT|132** [said](https://example.com/link):
````quote
> original question

Previous bot response here...
------
Tokens: 1000 (input) + 100 (output) = 1100
````

What is an NFA?'''
        
        # Strip mention
        prompt = re.sub(mention_pattern, "", message).strip()
        
        # Strip quote block
        quote_pattern = r'\[said\]\([^)]+\):\s*(`{3,})quote\s.*?\1'
        prompt = re.sub(quote_pattern, "", prompt, flags=re.DOTALL).strip()
        
        assert prompt == "What is an NFA?"
    
    def test_nested_quote_reply(self, mention_pattern):
        """Test stripping nested quote blocks."""
        message = '''@_**ChatGPT|132** [said](https://example.com/link):
````quote
@_**ChatGPT|132** [said](https://example.com/other):
```quote
> first question
First response
```
Second question
````

Third question here.'''
        
        # Strip ONLY FIRST mention (count=1 to preserve inner mentions)
        original_message = re.sub(mention_pattern, "", message, count=1).strip()
        
        # Inner ChatGPT mention should be preserved
        assert "@_**ChatGPT|132** [said]" in original_message
        
        # Strip quote block for prompt
        quote_pattern = r'\[said\]\([^)]+\):\s*(`{3,})quote\s.*?\1'
        prompt = re.sub(quote_pattern, "", original_message, flags=re.DOTALL).strip()
        
        assert prompt == "Third question here."
    
    def test_preserve_inner_bot_mentions(self, mention_pattern):
        """Test that bot mentions inside quotes are preserved."""
        message = '''@**ChatGPT** @_**jinguoliu|8** [said](https://example.com):
````quote
@_**ChatGPT|132** [said](https://example.com/prev):
```quote
> original question
Bot response
```
Follow up
````

New question'''
        
        # Strip only first bot mention
        original_message = re.sub(mention_pattern, "", message, count=1).strip()
        
        # Outer user attribution preserved
        assert "@_**jinguoliu|8** [said]" in original_message
        # Inner bot mention preserved
        assert "@_**ChatGPT|132** [said]" in original_message
    
    def test_user_quote_with_bot_mention(self, mention_pattern):
        """Test when user quotes another user's message that contains bot mention."""
        # User quotes their own previous message that mentioned the bot
        message = '''@**ChatGPT** @_**jinguoliu|8** [said](https://zulip.hkust-gz.edu.cn/#narrow/channel/128-Teaching-DSAA3071-2026-Spring/topic/GPT.20test/near/95980):
```quote
@**ChatGPT** You are so good.
```

What is a DFA?'''
        
        # Strip only the first bot mention (the one addressing the bot)
        prompt = re.sub(mention_pattern, "", message, count=1).strip()
        
        # The user attribution and inner bot mention should be preserved
        assert "@_**jinguoliu|8** [said]" in prompt
        
        # Strip quote block to get the actual question
        quote_pattern = r'@_\*\*[^*]+\*\*\s*\[said\]\([^)]+\):\s*(`{3,})quote\s.*?\1'
        prompt = re.sub(quote_pattern, "", prompt, flags=re.DOTALL).strip()
        
        assert prompt == "What is a DFA?"


# =============================================================================
# Response Formatting Tests
# =============================================================================

class TestResponseFormatting:
    """Test response formatting with quotes and mentions."""
    
    def test_quote_message_simple(self):
        """Test quoting a simple message."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            result = bot._quote_message("Hello world")
            assert result == "````quote\nHello world\n````"
    
    def test_quote_message_with_triple_backticks(self):
        """Test quoting a message containing triple backticks."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            message = "```quote\ninner content\n```"
            result = bot._quote_message(message)
            # Should use 4 backticks since content has 3
            assert result.startswith("````quote\n")
            assert result.endswith("\n````")
            assert "```quote" in result
    
    def test_quote_message_with_four_backticks(self):
        """Test quoting a message containing four backticks."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            message = "````quote\ninner content\n````"
            result = bot._quote_message(message)
            # Should use 5 backticks since content has 4
            assert result.startswith("`````quote\n")
            assert result.endswith("\n`````")
    
    def test_quote_message_nested(self):
        """Test quoting a nested quote message."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            message = '''@_**user|123** [said](url):
````quote
@_**bot|456** [said](url2):
```quote
original
```
reply
````

new question'''
            result = bot._quote_message(message)
            # Should use 5 backticks since content has 4
            assert result.startswith("`````quote\n")
            assert result.endswith("\n`````")
            # Content preserved
            assert "@_**user|123**" in result
            assert "@_**bot|456**" in result
    
    def test_format_stream_response_with_mention(self):
        """Test response formatting with user mention."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            
            # Mock usage object
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.total_tokens = 150
            
            result = bot._format_stream_response(
                question="What is DFA?",
                reply="A DFA is a deterministic finite automaton.",
                usage=usage,
                sender_name="jinguoliu",
                sender_id=8,
                message_url="#narrow/channel/128/topic/test/near/123"
            )
            
            # Check mention format
            assert "@_**jinguoliu|8** [said](#narrow/channel/128/topic/test/near/123):" in result
            # Check quote block
            assert "````quote" in result
            assert "What is DFA?" in result
            # Check reply
            assert "A DFA is a deterministic finite automaton." in result
            # Check tokens
            assert "Tokens: 100 (input) + 50 (output) = 150" in result
    
    def test_format_stream_response_without_url(self):
        """Test response formatting without message URL."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.total_tokens = 150
            
            result = bot._format_stream_response(
                question="Hello",
                reply="Hi there!",
                usage=usage,
                sender_name="user",
                sender_id=123
            )
            
            # Should have mention without [said](url)
            assert "@_**user|123**:" in result
            assert "[said]" not in result
    
    def test_format_stream_response_without_sender(self):
        """Test response formatting without sender info."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.total_tokens = 150
            
            result = bot._format_stream_response(
                question="Hello",
                reply="Hi there!",
                usage=usage
            )
            
            # Should not have mention
            assert "@_**" not in result
            # Should still have quote and reply
            assert "````quote" in result
            assert "Hello" in result
            assert "Hi there!" in result


# =============================================================================
# Generated File Tests
# =============================================================================

class TestGeneratedFiles:
    """Test generated file handling from CodeInterpreter."""
    
    def test_generated_file_dataclass(self):
        """Test GeneratedFile dataclass creation."""
        file = GeneratedFile(
            filename="plot.png",
            content=b"\x89PNG\r\n\x1a\n",  # PNG header bytes
            mime_type="image/png"
        )
        assert file.filename == "plot.png"
        assert file.content == b"\x89PNG\r\n\x1a\n"
        assert file.mime_type == "image/png"
    
    def test_generated_file_default_mime_type(self):
        """Test GeneratedFile default mime type."""
        file = GeneratedFile(
            filename="data.bin",
            content=b"binary data"
        )
        assert file.mime_type == "application/octet-stream"
    
    def test_file_uploader_callback(self):
        """Test that file_uploader callback is called correctly."""
        uploaded_files = []
        
        def mock_uploader(filename: str, content: bytes) -> str:
            uploaded_files.append((filename, content))
            return f"/uploads/{filename}"
        
        with patch('chatgpt.OpenAIClient') as MockClient, \
             patch('chatgpt.Runner') as MockRunner, \
             patch('chatgpt.SQLiteSession'), \
             patch('chatgpt.OpenAIResponsesCompactionSession') as MockSession:
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            MockSession.return_value = MagicMock()
            
            # Mock result with no files (simpler test)
            mock_result = MagicMock()
            mock_result.final_output = "Here's a plot."
            mock_result.raw_responses = [MagicMock()]
            mock_result.raw_responses[0].usage.input_tokens = 100
            mock_result.raw_responses[0].usage.output_tokens = 50
            mock_result.raw_responses[0].usage.total_tokens = 150
            mock_result.raw_responses[0].output = []  # No code interpreter output
            MockRunner.run_sync.return_value = mock_result
            
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                vector_store_id="vs_123",
                file_uploader=mock_uploader
            )
            
            # Test that bot was initialized with file_uploader
            assert bot.file_uploader == mock_uploader
            
            # Run a response (no files generated in this case)
            result = bot.get_dm_response("user@test.com", "Generate a plot")
            assert "Here's a plot" in result
    
    def test_extract_generated_files_empty(self):
        """Test file extraction when no files are generated."""
        with patch('chatgpt.OpenAIClient') as MockClient:
            MockClient.return_value = MagicMock()
            
            bot = ChatBot(model="gpt-4o", api_key="fake-key")
            
            # Mock result with no code interpreter output
            mock_result = MagicMock()
            mock_result.raw_responses = [MagicMock()]
            mock_result.raw_responses[0].output = []
            
            files = bot._extract_generated_files(mock_result)
            assert files == []
    
    def test_extract_generated_files_with_code_interpreter_output(self):
        """Test file extraction from code interpreter results."""
        with patch('chatgpt.OpenAIClient') as MockClient:
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            
            # Mock file content response
            mock_content_response = MagicMock()
            mock_content_response.read.return_value = b"fake image bytes"
            mock_client.containers.files.content.return_value = mock_content_response
            
            bot = ChatBot(model="gpt-4o", api_key="fake-key")
            
            # Mock result with code interpreter file output
            mock_file_info = MagicMock()
            mock_file_info.container_id = "cntr_123"
            mock_file_info.file_id = "file_456"
            mock_file_info.filename = "plot.png"
            mock_file_info.mime_type = "image/png"
            
            mock_code_result = MagicMock()
            mock_code_result.type = "files"
            mock_code_result.files = [mock_file_info]
            
            mock_code_item = MagicMock()
            mock_code_item.type = "code_interpreter_call"
            mock_code_item.results = [mock_code_result]
            
            mock_response = MagicMock()
            mock_response.output = [mock_code_item]
            
            mock_result = MagicMock()
            mock_result.raw_responses = [mock_response]
            
            files = bot._extract_generated_files(mock_result)
            
            assert len(files) == 1
            assert files[0].filename == "plot.png"
            assert files[0].content == b"fake image bytes"
            assert files[0].mime_type == "image/png"
            
            # Verify the container API was called
            mock_client.containers.files.content.assert_called_once_with(
                container_id="cntr_123",
                file_id="file_456"
            )
    
    def test_dm_response_with_generated_images(self):
        """Test DM response includes uploaded images."""
        uploaded_files = []
        
        def mock_uploader(filename: str, content: bytes) -> str:
            uploaded_files.append(filename)
            return f"/user_uploads/{filename}"
        
        with patch('chatgpt.OpenAIClient') as MockClient, \
             patch('chatgpt.Runner') as MockRunner, \
             patch('chatgpt.SQLiteSession'), \
             patch('chatgpt.OpenAIResponsesCompactionSession') as MockSession:
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            MockSession.return_value = MagicMock()
            
            # Mock file content download
            mock_content_response = MagicMock()
            mock_content_response.read.return_value = b"fake png bytes"
            mock_client.containers.files.content.return_value = mock_content_response
            
            # Build mock result with code interpreter file
            mock_file_info = MagicMock()
            mock_file_info.container_id = "cntr_abc"
            mock_file_info.file_id = "file_xyz"
            mock_file_info.filename = "chart.png"
            mock_file_info.mime_type = "image/png"
            
            mock_code_result = MagicMock()
            mock_code_result.type = "files"
            mock_code_result.files = [mock_file_info]
            
            mock_code_item = MagicMock()
            mock_code_item.type = "code_interpreter_call"
            mock_code_item.results = [mock_code_result]
            
            mock_response = MagicMock()
            mock_response.output = [mock_code_item]
            mock_response.usage.input_tokens = 200
            mock_response.usage.output_tokens = 100
            mock_response.usage.total_tokens = 300
            
            mock_result = MagicMock()
            mock_result.final_output = "Here is the chart you requested."
            mock_result.raw_responses = [mock_response]
            
            MockRunner.run_sync.return_value = mock_result
            
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                vector_store_id="vs_123",
                file_uploader=mock_uploader
            )
            
            result = bot.get_dm_response("user@test.com", "Create a bar chart", user_name="Test User")
            
            # Check that the image was uploaded
            assert "chart.png" in uploaded_files
            
            # Check that the response includes the image markdown
            # Zulip uses [filename](url) syntax, not ![filename](url)
            # Zulip auto-generates image previews for image links
            assert "[chart.png](/user_uploads/chart.png)" in result
            assert "Here is the chart you requested." in result
            assert "Tokens:" in result
    
    def test_dm_response_with_user_name_logging(self):
        """Test that DM response logs user name when provided."""
        with patch('chatgpt.OpenAIClient') as MockClient, \
             patch('chatgpt.Runner') as MockRunner, \
             patch('chatgpt.SQLiteSession'), \
             patch('chatgpt.OpenAIResponsesCompactionSession') as MockSession, \
             patch('chatgpt.logging') as mock_logging:
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            MockSession.return_value = MagicMock()
            
            mock_result = MagicMock()
            mock_result.final_output = "Response"
            mock_result.raw_responses = [MagicMock()]
            mock_result.raw_responses[0].usage.input_tokens = 100
            mock_result.raw_responses[0].usage.output_tokens = 50
            mock_result.raw_responses[0].usage.total_tokens = 150
            mock_result.raw_responses[0].output = []
            MockRunner.run_sync.return_value = mock_result
            
            bot = ChatBot(model="gpt-4o", api_key="fake-key", vector_store_id="vs_123")
            
            bot.get_dm_response("user@example.com", "Hello", user_name="John Doe")
            
            # Check that logging.info was called with user name
            calls = [str(c) for c in mock_logging.info.call_args_list]
            # At least one call should contain "John Doe"
            assert any("John Doe" in c for c in calls)


class TestFileInput:
    """Tests for user-uploaded file processing (images and PDFs)."""
    
    def test_extract_files_no_downloader(self):
        """Test that no files are extracted when file_downloader is not set."""
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=None
            )
            
            message = "Check this image /user_uploads/2/f8/abc123/test.png please"
            cleaned, files = bot._extract_files_from_message(message)
            
            assert cleaned == message  # Unchanged
            assert files == []
    
    def test_extract_files_no_files(self):
        """Test message with no files returns unchanged."""
        def mock_downloader(path):
            return b"data"
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "Just a plain text message"
            cleaned, files = bot._extract_files_from_message(message)
            
            assert cleaned == message
            assert files == []
    
    def test_extract_files_with_image(self):
        """Test extracting an image URL from a message."""
        downloaded_files = []
        
        def mock_downloader(path):
            downloaded_files.append(path)
            # Return fake PNG bytes
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "Check this image /user_uploads/2/f8/abc123/test.png please"
            cleaned, files = bot._extract_files_from_message(message, "test_user")
            
            # Verify the file was downloaded
            assert "/user_uploads/2/f8/abc123/test.png" in downloaded_files
            
            # Verify the message was cleaned
            assert "[attached image: test.png]" in cleaned
            assert "/user_uploads/" not in cleaned
            
            # Verify image content was extracted
            assert len(files) == 1
            assert files[0]["type"] == "input_image"
            assert files[0]["image_url"].startswith("data:image/png;base64,")
    
    def test_extract_files_multiple_images(self):
        """Test extracting multiple image URLs from a message."""
        def mock_downloader(path):
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "First: /user_uploads/2/a1/img1.png and second: /user_uploads/2/b2/img2.jpg"
            cleaned, files = bot._extract_files_from_message(message)
            
            assert len(files) == 2
            assert "[attached image: img1.png]" in cleaned
            assert "[attached image: img2.jpg]" in cleaned
    
    def test_extract_files_with_pdf(self):
        """Test extracting a PDF URL from a message."""
        downloaded_files = []
        
        def mock_downloader(path):
            downloaded_files.append(path)
            # Return fake PDF bytes
            return b"%PDF-1.4" + b"\x00" * 100
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "Check this document /user_uploads/2/f8/abc123/report.pdf please"
            cleaned, files = bot._extract_files_from_message(message, "test_user")
            
            # Verify the file was downloaded
            assert "/user_uploads/2/f8/abc123/report.pdf" in downloaded_files
            
            # Verify the message was cleaned
            assert "[attached document: report.pdf]" in cleaned
            assert "/user_uploads/" not in cleaned
            
            # Verify PDF content was extracted with correct format
            assert len(files) == 1
            assert files[0]["type"] == "input_file"
            assert files[0]["filename"] == "report.pdf"
            assert files[0]["file_data"].startswith("data:application/pdf;base64,")
    
    def test_extract_files_mixed_image_and_pdf(self):
        """Test extracting both image and PDF from a message."""
        def mock_downloader(path):
            if path.endswith('.pdf'):
                return b"%PDF-1.4" + b"\x00" * 50
            else:
                return b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "Image: /user_uploads/2/a1/photo.png and PDF: /user_uploads/2/b2/doc.pdf"
            cleaned, files = bot._extract_files_from_message(message)
            
            # Both should be extracted
            assert len(files) == 2
            
            # Check image
            image_file = next(f for f in files if f["type"] == "input_image")
            assert image_file["image_url"].startswith("data:image/png;base64,")
            
            # Check PDF
            pdf_file = next(f for f in files if f["type"] == "input_file")
            assert pdf_file["filename"] == "doc.pdf"
            assert pdf_file["file_data"].startswith("data:application/pdf;base64,")
            
            # Check cleaned message
            assert "[attached image: photo.png]" in cleaned
            assert "[attached document: doc.pdf]" in cleaned
    
    def test_extract_files_with_various_documents(self):
        """Test extracting various document types (docx, csv, txt, etc.)."""
        downloaded = []
        
        def mock_downloader(path):
            downloaded.append(path)
            return b"document content"
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "Word: /user_uploads/2/a1/report.docx CSV: /user_uploads/2/a1/data.csv Text: /user_uploads/2/a1/notes.txt"
            cleaned, files = bot._extract_files_from_message(message)
            
            # All should be downloaded
            assert len(downloaded) == 3
            assert "/user_uploads/2/a1/report.docx" in downloaded
            assert "/user_uploads/2/a1/data.csv" in downloaded
            assert "/user_uploads/2/a1/notes.txt" in downloaded
            
            # All should be extracted as input_file
            assert len(files) == 3
            for f in files:
                assert f["type"] == "input_file"
                assert "filename" in f
                assert "file_data" in f
            
            # Check MIME types
            docx_file = next(f for f in files if f["filename"] == "report.docx")
            assert "vnd.openxmlformats" in docx_file["file_data"]
            
            csv_file = next(f for f in files if f["filename"] == "data.csv")
            assert "text/csv" in csv_file["file_data"]
            
            txt_file = next(f for f in files if f["filename"] == "notes.txt")
            assert "text/plain" in txt_file["file_data"]
            
            # Check cleaned message
            assert "[attached document: report.docx]" in cleaned
            assert "[attached document: data.csv]" in cleaned
            assert "[attached document: notes.txt]" in cleaned
    
    def test_extract_files_skip_unsupported(self):
        """Test that unsupported file types are skipped."""
        downloaded = []
        
        def mock_downloader(path):
            downloaded.append(path)
            return b"data"
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            # .zip and .exe are not supported
            message = "See /user_uploads/2/a1/archive.zip and /user_uploads/2/a1/img.png and /user_uploads/2/a1/app.exe"
            cleaned, files = bot._extract_files_from_message(message)
            
            # Only PNG should be downloaded
            assert "/user_uploads/2/a1/img.png" in downloaded
            assert "/user_uploads/2/a1/archive.zip" not in downloaded
            assert "/user_uploads/2/a1/app.exe" not in downloaded
            
            # Only one file should be extracted
            assert len(files) == 1
            
            # Unsupported URLs should remain in message
            assert "/user_uploads/2/a1/archive.zip" in cleaned
            assert "/user_uploads/2/a1/app.exe" in cleaned
    
    def test_extract_files_download_failure(self):
        """Test handling of download failures."""
        def mock_downloader(path):
            return None  # Simulate download failure
        
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                file_downloader=mock_downloader
            )
            
            message = "Check /user_uploads/2/a1/test.png"
            cleaned, files = bot._extract_files_from_message(message)
            
            # No files extracted on failure
            assert files == []
            # Original URL should remain (not replaced)
            assert "/user_uploads/2/a1/test.png" in cleaned
    
    def test_dm_response_with_input_image(self):
        """Test DM response handles user-uploaded images (multimodal input)."""
        def mock_downloader(path):
            return b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        
        with patch('chatgpt.OpenAIClient') as MockClient, \
             patch('chatgpt.Runner') as MockRunner, \
             patch('chatgpt.SQLiteSession'), \
             patch('chatgpt.OpenAIResponsesCompactionSession') as MockSession:
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            MockSession.return_value = MagicMock()
            
            mock_result = MagicMock()
            mock_result.final_output = "I can see the image you uploaded. It shows a graph."
            mock_result.raw_responses = [MagicMock()]
            mock_result.raw_responses[0].usage.input_tokens = 500
            mock_result.raw_responses[0].usage.output_tokens = 100
            mock_result.raw_responses[0].usage.total_tokens = 600
            mock_result.raw_responses[0].output = []
            MockRunner.run_sync.return_value = mock_result
            
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                vector_store_id="vs_123",
                file_downloader=mock_downloader
            )
            
            result = bot.get_dm_response(
                "user@test.com", 
                "What's in this image? /user_uploads/2/f8/abc/chart.png",
                user_name="Test User"
            )
            
            # Verify Runner.run_sync was called with multimodal input
            call_args = MockRunner.run_sync.call_args
            agent_input = call_args[0][1]  # Second positional arg is input
            
            # Should be a list (multimodal input)
            assert isinstance(agent_input, list)
            assert len(agent_input) == 1
            assert agent_input[0]["role"] == "user"
            
            content = agent_input[0]["content"]
            # Should have text + image parts
            assert any(c["type"] == "input_text" for c in content)
            assert any(c["type"] == "input_image" for c in content)
            
            # The text part should have the cleaned message
            text_part = next(c for c in content if c["type"] == "input_text")
            assert "[attached image: chart.png]" in text_part["text"]
            
            # Response should include the AI's analysis
            assert "I can see the image" in result
    
    def test_dm_response_with_input_pdf(self):
        """Test DM response handles user-uploaded PDFs (multimodal input)."""
        def mock_downloader(path):
            return b"%PDF-1.4" + b"\x00" * 50
        
        with patch('chatgpt.OpenAIClient') as MockClient, \
             patch('chatgpt.Runner') as MockRunner, \
             patch('chatgpt.SQLiteSession'), \
             patch('chatgpt.OpenAIResponsesCompactionSession') as MockSession:
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            MockSession.return_value = MagicMock()
            
            mock_result = MagicMock()
            mock_result.final_output = "I've analyzed the PDF. It contains course notes on automata."
            mock_result.raw_responses = [MagicMock()]
            mock_result.raw_responses[0].usage.input_tokens = 1000
            mock_result.raw_responses[0].usage.output_tokens = 150
            mock_result.raw_responses[0].usage.total_tokens = 1150
            mock_result.raw_responses[0].output = []
            MockRunner.run_sync.return_value = mock_result
            
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                vector_store_id="vs_123",
                file_downloader=mock_downloader
            )
            
            result = bot.get_dm_response(
                "user@test.com", 
                "Summarize this document: /user_uploads/2/f8/abc/notes.pdf",
                user_name="Test User"
            )
            
            # Verify Runner.run_sync was called with multimodal input
            call_args = MockRunner.run_sync.call_args
            agent_input = call_args[0][1]
            
            # Should be a list (multimodal input)
            assert isinstance(agent_input, list)
            assert len(agent_input) == 1
            assert agent_input[0]["role"] == "user"
            
            content = agent_input[0]["content"]
            # Should have text + file parts
            assert any(c["type"] == "input_text" for c in content)
            assert any(c["type"] == "input_file" for c in content)
            
            # Check the PDF was correctly formatted
            pdf_part = next(c for c in content if c["type"] == "input_file")
            assert pdf_part["filename"] == "notes.pdf"
            assert pdf_part["file_data"].startswith("data:application/pdf;base64,")
            
            # The text part should have the cleaned message
            text_part = next(c for c in content if c["type"] == "input_text")
            assert "[attached document: notes.pdf]" in text_part["text"]
            
            # Response should include the AI's analysis
            assert "I've analyzed the PDF" in result
    
    def test_dm_response_without_files_uses_string_input(self):
        """Test DM response uses simple string input when no files present."""
        with patch('chatgpt.OpenAIClient') as MockClient, \
             patch('chatgpt.Runner') as MockRunner, \
             patch('chatgpt.SQLiteSession'), \
             patch('chatgpt.OpenAIResponsesCompactionSession') as MockSession:
            
            mock_client = MagicMock()
            MockClient.return_value = mock_client
            MockSession.return_value = MagicMock()
            
            mock_result = MagicMock()
            mock_result.final_output = "Simple response"
            mock_result.raw_responses = [MagicMock()]
            mock_result.raw_responses[0].usage.input_tokens = 100
            mock_result.raw_responses[0].usage.output_tokens = 50
            mock_result.raw_responses[0].usage.total_tokens = 150
            mock_result.raw_responses[0].output = []
            MockRunner.run_sync.return_value = mock_result
            
            bot = ChatBot(
                model="gpt-4o",
                api_key="fake-key",
                vector_store_id="vs_123"
            )
            
            bot.get_dm_response("user@test.com", "Hello, no files here")
            
            # Verify Runner.run_sync was called with string input (not list)
            call_args = MockRunner.run_sync.call_args
            agent_input = call_args[0][1]
            
            assert isinstance(agent_input, str)
            assert agent_input == "Hello, no files here"
