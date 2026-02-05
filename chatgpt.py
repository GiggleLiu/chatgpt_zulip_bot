# chatgpt.py
"""
OpenAI-powered course assistant for DSAA3071 Theory of Computation.
Uses the OpenAI Agents SDK for automatic context management.

Modes:
- Stream: Responses API with RAG (vector store search, no chaining)
- DM: Agents SDK with auto-compaction (automatic context compression)
"""

from openai import OpenAI as OpenAIClient
import tiktoken
import logging
import os
import glob
import fnmatch
import re
import asyncio
import base64
from typing import Callable, Optional
from dataclasses import dataclass

# OpenAI Agents SDK imports
from agents import Agent, Runner, SQLiteSession, FileSearchTool, WebSearchTool, CodeInterpreterTool
from agents.memory import OpenAIResponsesCompactionSession

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_SPECS = {
    # GPT-5.x series
    "gpt-5.2": (400_000, 32_768, "cl100k_base"),
    "gpt-5.1": (400_000, 32_768, "cl100k_base"),
    "gpt-5": (400_000, 32_768, "cl100k_base"),
    # GPT-4.1 series  
    "gpt-4.1": (1_047_576, 32_768, "cl100k_base"),
    "gpt-4.1-mini": (1_047_576, 32_768, "cl100k_base"),
    # GPT-4o series
    "gpt-4o": (128_000, 16_384, "o200k_base"),
    "gpt-4o-mini": (128_000, 16_384, "o200k_base"),
    "gpt-4o-2024-11-20": (128_000, 16_384, "o200k_base"),
    "gpt-4o-2024-08-06": (128_000, 16_384, "o200k_base"),
    "gpt-4o-2024-05-13": (128_000, 4_096, "o200k_base"),
    # GPT-4 Turbo
    "gpt-4-turbo": (128_000, 4_096, "cl100k_base"),
    "gpt-4-turbo-preview": (128_000, 4_096, "cl100k_base"),
    # GPT-4 base
    "gpt-4": (8_192, 4_096, "cl100k_base"),
    "gpt-4-32k": (32_768, 4_096, "cl100k_base"),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": (16_385, 4_096, "cl100k_base"),
    "gpt-3.5-turbo-16k": (16_385, 4_096, "cl100k_base"),
    # o-series (reasoning models)
    "o1": (200_000, 100_000, "o200k_base"),
    "o1-mini": (128_000, 65_536, "o200k_base"),
    "o1-preview": (128_000, 32_768, "o200k_base"),
    "o3": (200_000, 100_000, "o200k_base"),
    "o3-mini": (200_000, 100_000, "o200k_base"),
    "o4-mini": (200_000, 100_000, "o200k_base"),
}

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_OUTPUT = 16_384
DEFAULT_ENCODING = "cl100k_base"


def get_model_specs(model: str) -> tuple[int, int, str]:
    """Get model specifications (context_window, max_output_tokens, encoding)."""
    model_lower = model.lower()
    
    if model_lower in MODEL_SPECS:
        return MODEL_SPECS[model_lower]
    
    for known_model in sorted(MODEL_SPECS.keys(), key=len, reverse=True):
        if model_lower.startswith(known_model):
            return MODEL_SPECS[known_model]
    
    if "gpt-5" in model_lower:
        return (400_000, 32_768, "cl100k_base")
    elif "gpt-4.1" in model_lower:
        return (1_047_576, 32_768, "cl100k_base")
    elif "gpt-4o" in model_lower:
        return (128_000, 16_384, "o200k_base")
    elif "gpt-4" in model_lower:
        return (128_000, 4_096, "cl100k_base")
    elif "gpt-3.5" in model_lower:
        return (16_385, 4_096, "cl100k_base")
    elif model_lower.startswith(("o1", "o3", "o4")):
        return (200_000, 100_000, "o200k_base")
    
    logging.warning(f"Unknown model '{model}', using defaults")
    return (DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_OUTPUT, DEFAULT_ENCODING)


# =============================================================================
# Course Materials
# =============================================================================

DEFAULT_COURSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DSAA3071TheoryOfComputation")

# Command patterns
RESET_PATTERN = re.compile(r'^/reset\b', re.IGNORECASE)

# Zulip file URL pattern - matches /user_uploads/{realm_id}/{path}
# Example: /user_uploads/2/f8/QqqhmDJLNixmmb2EaTbzI008/origin.png
ZULIP_FILE_URL_PATTERN = re.compile(r'/user_uploads/\d+/[a-zA-Z0-9_/-]+\.[a-zA-Z0-9]+')

# Supported file extensions for multimodal input
# Images: passed as input_image with base64 image_url
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
# Documents: passed as input_file with base64 file_data
# PDF: OpenAI extracts text + renders pages as images
# Other formats: OpenAI may support these (let API reject if not)
DOCUMENT_EXTENSIONS = {
    '.pdf',      # PDF documents (confirmed supported)
    '.docx',     # Microsoft Word
    '.doc',      # Microsoft Word (legacy)
    '.xlsx',     # Microsoft Excel
    '.xls',      # Microsoft Excel (legacy)
    '.pptx',     # Microsoft PowerPoint
    '.ppt',      # Microsoft PowerPoint (legacy)
    '.csv',      # Comma-separated values
    '.txt',      # Plain text
    '.md',       # Markdown
    '.json',     # JSON data
    '.html',     # HTML documents
    '.xml',      # XML documents
}

# MIME types for document extensions
DOCUMENT_MIME_TYPES = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.ppt': 'application/vnd.ms-powerpoint',
    '.csv': 'text/csv',
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.json': 'application/json',
    '.html': 'text/html',
    '.xml': 'application/xml',
}


@dataclass
class GeneratedFile:
    """A file generated by CodeInterpreter."""
    filename: str
    content: bytes
    mime_type: str = "application/octet-stream"


def load_course_materials(course_dir: str = None) -> dict:
    """Load all typst files from the course directory, organized by week."""
    if course_dir is None:
        course_dir = DEFAULT_COURSE_DIR
    
    materials = {}
    
    if not os.path.exists(course_dir):
        logging.warning(f"Course directory not found: {course_dir}")
        return materials
    
    typ_files = glob.glob(os.path.join(course_dir, "**/*.typ"), recursive=True)
    
    for filepath in typ_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            rel_path = os.path.relpath(filepath, course_dir)
            parts = rel_path.split(os.sep)
            category = parts[0] if parts[0].startswith('week') else 'other'
            
            if category not in materials:
                materials[category] = {}
            materials[category][rel_path] = content
                
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
    
    return materials


def extract_text_content(typst_content: str) -> str:
    """Extract readable text from typst content, removing heavy formatting."""
    skip_patterns = ['#import', '#let ', '#show', '#align', '#box(', '#rect(', '#table(', 
                     '#canvas(', '#diagram(', '#grid(', '#place(', 'gradient.', 'rgb(']
    
    lines = []
    for line in typst_content.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
        if any(stripped.startswith(p) for p in skip_patterns):
            continue
        if stripped.startswith('=') or stripped.startswith('-') or stripped.startswith('*'):
            lines.append(stripped)
        elif stripped.startswith('[') and stripped.endswith(']'):
            lines.append(stripped[1:-1])
        elif not stripped.startswith('#') and not stripped.startswith(')'):
            lines.append(stripped)
    
    return '\n'.join(lines)


def match_file_pattern(filename: str, patterns: list) -> bool:
    """Check if filename matches any of the patterns."""
    for pattern in patterns:
        if any(c in pattern for c in ['*', '?', '[']):
            if fnmatch.fnmatch(filename, pattern):
                return True
        else:
            if pattern in filename:
                return True
    return False


def get_week_context(materials: dict, week_num: int, file_patterns: list = None, max_chars: int = 300_000) -> str:
    """Get course context for a specific week."""
    week_key = f"week{week_num}"
    if week_key not in materials:
        return ""
    
    context_parts = [f"=== WEEK {week_num} COURSE MATERIALS ===\n"]
    
    for filename, content in materials[week_key].items():
        if file_patterns and not match_file_pattern(filename, file_patterns):
            continue
        
        clean_content = extract_text_content(content)
        if clean_content.strip():
            context_parts.append(f"--- {filename} ---\n{clean_content}")
    
    full_context = "\n".join(context_parts)
    return full_context[:max_chars]


# =============================================================================
# ChatBot Class
# =============================================================================

class ChatBot:
    """
    OpenAI-powered chatbot using the Agents SDK.
    
    Modes:
    - Stream messages: Responses API with RAG (vector store search, no chaining)
    - DM messages: Agents SDK with auto-compaction (automatic context compression)
    """
    
    def __init__(self, model: str, api_key: str, course_dir: str = None, 
                 file_patterns: list = None, vector_store_id: str = None,
                 max_output_tokens: int = None, log_qa: bool = False,
                 session_db_path: str = "conversations.db",
                 file_uploader: Optional[Callable[[str, bytes], str]] = None,
                 file_downloader: Optional[Callable[[str], bytes | None]] = None):
        """
        Initialize the chatbot.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            course_dir: Path to course materials directory (for week detection)
            file_patterns: Patterns to filter course files
            vector_store_id: OpenAI Vector Store ID for RAG
            max_output_tokens: Override for max output tokens
            log_qa: Whether to log each Q&A pair
            session_db_path: Path to SQLite database for session storage
            file_uploader: Callback to upload files, takes (filename, bytes) returns URL
            file_downloader: Callback to download files from Zulip, takes (path) returns bytes or None
        """
        self.model = model
        self.client = OpenAIClient(api_key=api_key)
        self.file_uploader = file_uploader
        self.file_downloader = file_downloader
        
        # Set API key for Agents SDK (uses env var)
        os.environ["OPENAI_API_KEY"] = api_key
        
        _, default_max_output, encoding_name = get_model_specs(model)
        self.max_output_tokens = max_output_tokens or default_max_output
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # OpenAI resources
        self.vector_store_id = vector_store_id
        
        # Logging settings
        self.log_qa = log_qa
        
        # Session storage for DM mode (Agents SDK with compaction)
        self.session_db_path = session_db_path
        self.user_sessions = {}  # user_id -> OpenAIResponsesCompactionSession
        
        # Create Agent for DM mode (with file search tool)
        self.dm_agent = self._create_dm_agent() if vector_store_id else None
        
        # Course materials (for week detection)
        self.file_patterns = file_patterns or []
        self.course_materials = load_course_materials(course_dir)
        
        # Available weeks
        self.available_weeks = sorted([
            int(k.replace('week', '')) 
            for k in self.course_materials.keys() 
            if k.startswith('week')
        ])
        
        logging.info(f"ChatBot initialized: model={model}")
        logging.info(f"Using Agents SDK with auto-compaction for DM mode")
        logging.info(f"Available weeks: {self.available_weeks}")
        if vector_store_id:
            logging.info(f"Vector store: {vector_store_id}")
    
    def _create_dm_agent(self) -> Agent:
        """Create an Agent for DM mode with file search, web search, and code interpreter."""
        # Use identical instructions as stream mode for consistency
        instructions = self._get_base_instructions() + """

=== USER-UPLOADED FILES ===

When users upload files (images, PDFs, documents), the content is provided DIRECTLY in the message.
You can see and analyze uploaded files immediately - no tool is needed to access them.
DO NOT use file_search for user uploads - file_search only searches the course materials database.

=== AVAILABLE TOOLS ===

1. **file_search**: Search COURSE MATERIALS ONLY (lecture notes, assignments, etc. in the vector store).
   - Use for course-related questions about Theory of Computation topics

2. **web_search**: Search the web for current information, papers, documentation, or concepts not covered in course materials.

3. **code_interpreter**: Execute Python code in a sandbox. Use this to:
   - Run automata simulations or regex examples
   - Process or transform data from user uploads
   - Create visualizations or diagrams
   - Perform mathematical computations

For course questions: use file_search first, then web_search if needed.
For user uploads: analyze directly from the message content."""
        
        return Agent(
            name="Professor-GPT",
            model=self.model,
            instructions=instructions,
            tools=[
                FileSearchTool(
                    vector_store_ids=[self.vector_store_id],
                    max_num_results=5,
                ),
                WebSearchTool(),
                CodeInterpreterTool(tool_config={
                    "type": "code_interpreter",
                    "container": {"type": "auto"}
                }),
            ],
        )
    
    def _get_user_session(self, user_id: str) -> OpenAIResponsesCompactionSession:
        """Get or create a compaction session for a user."""
        if user_id not in self.user_sessions:
            # Create underlying SQLite session for persistence
            underlying = SQLiteSession(user_id, self.session_db_path)
            
            # Wrap with compaction session for automatic history compression
            self.user_sessions[user_id] = OpenAIResponsesCompactionSession(
                session_id=user_id,
                underlying_session=underlying,
            )
            logging.info(f"Created compaction session for user {user_id}")
        
        return self.user_sessions[user_id]
    
    def _extract_generated_files(self, result) -> list[GeneratedFile]:
        """Extract files generated by CodeInterpreter from the agent result.
        
        Looks for container_file_citation annotations and downloads the files.
        """
        files = []
        
        try:
            # Check raw_responses for code interpreter outputs
            for response in (result.raw_responses or []):
                # Look through output items
                for item in (getattr(response, 'output', None) or []):
                    # Check for code_interpreter_call items
                    if getattr(item, 'type', None) == 'code_interpreter_call':
                        # Get results from the code interpreter call
                        for code_result in (getattr(item, 'results', None) or []):
                            # Check for file outputs
                            if getattr(code_result, 'type', None) == 'files':
                                for file_info in (getattr(code_result, 'files', None) or []):
                                    container_id = getattr(file_info, 'container_id', None)
                                    file_id = getattr(file_info, 'file_id', None)
                                    filename = getattr(file_info, 'filename', None) or f"{file_id}.png"
                                    mime_type = getattr(file_info, 'mime_type', None) or 'image/png'
                                    
                                    if container_id and file_id:
                                        try:
                                            # Download file content from OpenAI container
                                            content = self.client.containers.files.content(
                                                container_id=container_id,
                                                file_id=file_id
                                            )
                                            # content is a Response object, get bytes
                                            file_bytes = content.read()
                                            files.append(GeneratedFile(
                                                filename=filename,
                                                content=file_bytes,
                                                mime_type=mime_type
                                            ))
                                            logging.info(f"Downloaded generated file: {filename}")
                                        except Exception as e:
                                            logging.error(f"Failed to download file {file_id}: {e}")
                    
                    # Also check message items for annotations
                    if getattr(item, 'type', None) == 'message':
                        for content in (getattr(item, 'content', None) or []):
                            for ann in (getattr(content, 'annotations', None) or []):
                                if getattr(ann, 'type', None) == 'container_file_citation':
                                    container_id = getattr(ann, 'container_id', None)
                                    file_id = getattr(ann, 'file_id', None)
                                    filename = getattr(ann, 'filename', None) or f"{file_id}.png"
                                    
                                    if container_id and file_id:
                                        try:
                                            content_resp = self.client.containers.files.content(
                                                container_id=container_id,
                                                file_id=file_id
                                            )
                                            file_bytes = content_resp.read()
                                            # Determine mime type from filename
                                            if filename.endswith('.png'):
                                                mime_type = 'image/png'
                                            elif filename.endswith(('.jpg', '.jpeg')):
                                                mime_type = 'image/jpeg'
                                            elif filename.endswith('.gif'):
                                                mime_type = 'image/gif'
                                            else:
                                                mime_type = 'application/octet-stream'
                                            
                                            files.append(GeneratedFile(
                                                filename=filename,
                                                content=file_bytes,
                                                mime_type=mime_type
                                            ))
                                            logging.info(f"Downloaded file from annotation: {filename}")
                                        except Exception as e:
                                            logging.error(f"Failed to download annotated file {file_id}: {e}")
        except Exception as e:
            logging.error(f"Error extracting generated files: {e}")
        
        return files
    
    def _get_base_instructions(self) -> str:
        """Base instructions for AI assistant."""
        return """You are a university professor teaching DSAA3071 Theory of Computation at HKUST(GZ).

Your role is to:
- Help students understand concepts in automata theory, formal languages, and computability
- Explain DFA, NFA, regular expressions, context-free grammars, pushdown automata, and Turing machines
- Guide students through proofs and problem-solving techniques
- Be concise but thorough in explanations

As a professor, you occasionally:
- Draw connections to related topics (complexity theory, algorithms, programming languages, logic)
- Share insights from your professional knowledge to inspire deeper thinking
- Ask thought-provoking questions that encourage students to explore further
- Mention real-world applications or historical context when relevant

=== ACCURACY IS YOUR TOP PRIORITY ===

CRITICAL RULES:
1. Only answer questions you are sure about. If you are not certain, use web search or encourage students to ask human professors.

=== MATH FORMATTING FOR ZULIP ===

INLINE MATH (within text):
Use $$ with SPACES around them.
- "The function $$f(x)$$ is defined"
- "where $$\\delta$$ is the transition function"

BLOCK MATH (standalone equations):
Use fenced code blocks with "math" language:
```math
f(x) = x^2
```

```math
\\delta: Q \\times \\Sigma \\to Q
```

WRONG (DO NOT DO THIS):
- `$$f(x)$$` - NO backticks around inline math!
- $$
f(x)
$$ - NO block math with double dollars!
- "function$$f(x)$$is" - NO missing spaces for inline!
=== END MATH FORMATTING ==="""
    
    def _quote_message(self, text: str) -> str:
        """Quote a message using Zulip markdown.
        
        Uses quote block syntax with enough backticks to handle nested code blocks.
        """
        # Find the longest sequence of backticks in the text
        max_backticks = 0
        current_count = 0
        for char in text:
            if char == '`':
                current_count += 1
                max_backticks = max(max_backticks, current_count)
            else:
                current_count = 0
        
        # Use one more backtick than the max found (minimum 4 for quote blocks)
        fence = '`' * max(4, max_backticks + 1)
        return f"{fence}quote\n{text}\n{fence}"
    
    def _format_stream_response(self, question: str, reply: str, usage, sender_name: str = None, sender_id: int = None, message_url: str = None) -> str:
        """Format stream response with user mention, quoted question, and usage info."""
        quoted_question = self._quote_message(question)
        
        # Add user mention in Zulip quote format: @_**Name|ID** [said](url):
        if sender_name and sender_id and message_url:
            mention = f"@_**{sender_name}|{sender_id}** [said]({message_url}):\n"
        elif sender_name and sender_id:
            mention = f"@_**{sender_name}|{sender_id}**:\n"
        else:
            mention = ""
        
        return (
            f"{mention}{quoted_question}\n\n"
            f"{reply}\n"
            f"------\n"
            f"Tokens: {usage.input_tokens:,} (input) + {usage.output_tokens:,} (output) "
            f"= {usage.total_tokens:,}"
        )
    
    def _format_dm_response(self, reply: str, usage) -> str:
        """Format DM response with just reply and usage info (no quoting needed)."""
        return (
            f"{reply}\n"
            f"------\n"
            f"Tokens: {usage.input_tokens:,} (input) + {usage.output_tokens:,} (output) "
            f"= {usage.total_tokens:,}"
        )
    
    def _extract_files_from_message(self, message: str, user_display: str = None) -> tuple[str, list[dict]]:
        """Extract Zulip file URLs from a message and download them.
        
        Supports both images and PDF documents.
        
        Returns:
            tuple: (cleaned_message, list of content dicts for multimodal input)
            
        Content formats:
            Images: {"type": "input_image", "image_url": "data:image/png;base64,..."}
            PDFs:   {"type": "input_file", "filename": "...", "file_data": "data:application/pdf;base64,..."}
        """
        if not self.file_downloader:
            return message, []
        
        # Find all Zulip file URLs in the message
        file_urls = ZULIP_FILE_URL_PATTERN.findall(message)
        if not file_urls:
            return message, []
        
        file_contents = []
        cleaned_message = message
        
        for file_url in file_urls:
            ext = os.path.splitext(file_url)[1].lower()
            filename = os.path.basename(file_url)
            
            # Check if it's a supported file type
            is_image = ext in IMAGE_EXTENSIONS
            is_document = ext in DOCUMENT_EXTENSIONS
            
            if not is_image and not is_document:
                logging.info(f"[{user_display}] Skipping unsupported file type: {file_url}")
                continue
            
            # Download the file
            try:
                file_bytes = self.file_downloader(file_url)
                if file_bytes:
                    b64_data = base64.b64encode(file_bytes).decode('utf-8')
                    
                    if is_image:
                        # Images use input_image format
                        mime_type = {
                            '.png': 'image/png',
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.gif': 'image/gif',
                            '.webp': 'image/webp',
                        }.get(ext, 'image/png')
                        
                        file_contents.append({
                            "type": "input_image",
                            "image_url": f"data:{mime_type};base64,{b64_data}",
                        })
                        file_type_label = "image"
                    else:
                        # Documents use input_file format
                        mime_type = DOCUMENT_MIME_TYPES.get(ext, 'application/octet-stream')
                        
                        file_contents.append({
                            "type": "input_file",
                            "filename": filename,
                            "file_data": f"data:{mime_type};base64,{b64_data}",
                        })
                        file_type_label = "document"
                    
                    logging.info(f"[{user_display}] Downloaded and encoded {file_type_label}: {file_url} ({len(file_bytes)} bytes)")
                    
                    # Remove the URL from the message to avoid confusion
                    cleaned_message = cleaned_message.replace(file_url, f"[attached {file_type_label}: {filename}]")
                else:
                    logging.warning(f"[{user_display}] Failed to download file: {file_url}")
            except Exception as e:
                logging.error(f"[{user_display}] Error processing file {file_url}: {e}")
        
        return cleaned_message, file_contents
    
    def get_stream_response(self, user_id: str, prompt: str, original_message: str = None, sender_name: str = None, sender_id: int = None, message_url: str = None) -> str:
        """
        Handle stream message: RAG mode, no chaining.
        
        Uses vector store to search relevant content per query.
        
        Args:
            user_id: User identifier (email)
            prompt: The actual question to process (quote blocks stripped)
            original_message: Full message for quoting in response (optional, defaults to prompt)
            sender_name: Sender's display name for @mention in response
            sender_id: Sender's numeric ID for Zulip mention format
            message_url: URL to the original message for [said](url) format
        """
        logging.info(f"Stream message from {user_id}")
        
        # Use original_message for quoting if provided, otherwise use prompt
        quote_text = original_message if original_message else prompt
        
        if not self.vector_store_id:
            return (
                self._quote_message(quote_text) + "\n\n"
                f"**Error:** RAG not configured for stream messages.\n\n"
                f"Please set `VECTOR_STORE_ID` in config.ini (run `make upload` first)."
            )
        
        try:
            instructions = self._get_base_instructions() + """

=== USER-UPLOADED FILES ===

When users upload files (images, PDFs, documents), the content is provided DIRECTLY in the message.
You can see and analyze uploaded files immediately - no tool is needed to access them.
DO NOT use file_search for user uploads - file_search only searches the course materials database.

You have access to course materials through the file_search tool. 
Search for relevant content to answer course-related questions accurately."""
            
            # Extract and process any files (images, PDFs) from the message
            display_name = f"{sender_name} ({user_id})" if sender_name else user_id
            cleaned_prompt, file_contents = self._extract_files_from_message(prompt, display_name)
            
            # Build input for the API
            if file_contents:
                # Multimodal input: list of content items
                content_parts = [{"type": "input_text", "text": cleaned_prompt}]
                content_parts.extend(file_contents)
                api_input = [{"role": "user", "content": content_parts}]
                logging.info(f"[{display_name}] Stream: Using multimodal input with {len(file_contents)} file(s)")
            else:
                # Simple text input
                api_input = prompt
            
            response = self.client.responses.create(
                model=self.model,
                input=api_input,
                instructions=instructions,
                max_output_tokens=self.max_output_tokens,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id],
                    "max_num_results": 5,  # Match DM mode for consistency
                }],
                truncation="auto",
            )
            
            reply = response.output_text or "I couldn't generate a response."
            
            if self.log_qa:
                logging.info(f"[Stream Q&A] User={user_id}\nQ: {prompt}\nA: {reply}")
            
            return self._format_stream_response(quote_text, reply, response.usage, sender_name, sender_id, message_url)
            
        except Exception as e:
            logging.error(f"Stream response error: {e}")
            if sender_name and sender_id and message_url:
                mention = f"@_**{sender_name}|{sender_id}** [said]({message_url}):\n"
            elif sender_name and sender_id:
                mention = f"@_**{sender_name}|{sender_id}**:\n"
            else:
                mention = ""
            return mention + self._quote_message(quote_text) + f"\n\nError: {str(e)}"
    
    def get_dm_response(self, user_id: str, prompt: str, user_name: str = None) -> str:
        """
        Handle DM message using Agents SDK with auto-compaction.
        
        Uses OpenAIResponsesCompactionSession for automatic context compression.
        File search retrieves relevant course materials per query.
        Supports multimodal input (images) from Zulip uploads.
        
        Args:
            user_id: User identifier (email)
            prompt: The user's message
            user_name: Optional display name for logging
        """
        display_name = f"{user_name} ({user_id})" if user_name else user_id
        logging.info(f"DM message from {display_name}")
        
        prompt_stripped = prompt.strip()
        
        # Check for /reset command first - works even without full configuration
        if RESET_PATTERN.match(prompt_stripped):
            self._clear_user_session_sync(user_id)
            return "Conversation cleared. Your next message will start a fresh conversation."
        
        # Check if vector store is configured (needed for file search)
        if not self.vector_store_id or not self.dm_agent:
            return (
                "**Error:** Vector store not configured.\n\n"
                "Please run `make upload` to create a vector store, then add `VECTOR_STORE_ID` to config.ini."
            )
        
        try:
            # Get session for user (with auto-compaction)
            session = self._get_user_session(user_id)
            
            # Extract and process any files (images, PDFs) from the message
            cleaned_prompt, file_contents = self._extract_files_from_message(prompt, display_name)
            
            # Build input for the agent
            if file_contents:
                # Multimodal input: list of content items
                # Format: [{"role": "user", "content": [text_part, file_part, ...]}]
                content_parts = [{"type": "input_text", "text": cleaned_prompt}]
                content_parts.extend(file_contents)
                agent_input = [{"role": "user", "content": content_parts}]
                logging.info(f"[{display_name}] Using multimodal input with {len(file_contents)} file(s)")
            else:
                # Simple text input
                agent_input = prompt
            
            # Run agent with session (sync version)
            result = Runner.run_sync(
                self.dm_agent,
                agent_input,
                session=session,
            )
            
            reply = result.final_output or "I couldn't generate a response."
            
            # Extract and upload any generated files (images, etc.)
            generated_files = self._extract_generated_files(result)
            if generated_files and self.file_uploader:
                image_urls = []
                for gen_file in generated_files:
                    try:
                        url = self.file_uploader(gen_file.filename, gen_file.content)
                        if url:
                            image_urls.append((gen_file.filename, url))
                            logging.info(f"[{display_name}] Uploaded {gen_file.filename} -> {url}")
                    except Exception as e:
                        logging.error(f"[{display_name}] Failed to upload {gen_file.filename}: {e}")
                
                # Append images to reply using Zulip's link syntax
                # Zulip uses [filename](url) NOT ![filename](url)
                # and auto-generates image previews for image links
                if image_urls:
                    reply += "\n\n"
                    for filename, url in image_urls:
                        # Clean filename of [ and ] characters that break Markdown
                        clean_filename = filename.replace("[", "").replace("]", "")
                        reply += f"[{clean_filename}]({url})\n"
            
            if self.log_qa:
                logging.info(f"[DM Q&A] User={display_name}\nQ: {prompt}\nA: {reply}")
            
            # Format response with usage info
            usage = result.raw_responses[-1].usage if result.raw_responses else None
            if usage:
                return self._format_dm_response(reply, usage)
            else:
                return f"{reply}\n------\n(Token usage not available)"
            
        except Exception as e:
            logging.error(f"DM response error: {e}")
            # If session is broken, clear it so next message creates a new one
            if "session" in str(e).lower() or "conversation" in str(e).lower():
                self._clear_user_session_sync(user_id)
            return f"Error: {str(e)}"
    
    def _clear_user_session_sync(self, user_id: str):
        """Clear a user's session synchronously."""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            # Run async clear in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new loop for sync context
                    asyncio.run(session.clear_session())
                else:
                    loop.run_until_complete(session.clear_session())
            except RuntimeError:
                # No event loop, create one
                asyncio.run(session.clear_session())
            del self.user_sessions[user_id]
            logging.info(f"Cleared session for user {user_id}")
    
    def clear_user_session(self, user_id: str):
        """Clear a user's session."""
        self._clear_user_session_sync(user_id)


# Alias for backward compatibility
OpenAI = ChatBot
