# qwen.py
"""
Qwen-powered course assistant for DSAA3071 Theory of Computation.
Uses the Qwen-Agent framework for RAG and tool capabilities.

Two modes:
- STREAM (public): RAG with course materials via DocQA
- DM (private): Week-based context with conversation history

DM Flow:
1. Student DMs the bot
2. Bot prompts: "Use /week N to select a week"
3. Student types: /week 3
4. Bot loads Week 3 materials
5. Student asks questions with conversation history

Commands (DM only):
- /week N  - Select which week to study (e.g., /week 3)
- /reset   - Clear conversation and week selection

Multimodal Support:
- Images: PNG, JPEG, GIF, WebP (via Qwen-VL models)

Requirements:
- pip install -U "qwen-agent[rag]"

Documentation:
- https://qwenlm.github.io/Qwen-Agent/
- https://qwen.ai/apiplatform
"""

from qwen_agent.agents import Assistant
from qwen_agent.llm.schema import ContentItem
import base64
import logging
import os
import re
from typing import Callable, Optional

from chatgpt import (
    load_course_materials, 
    match_file_pattern, 
    extract_text_content,
    RESET_PATTERN,
    ZULIP_FILE_URL_PATTERN,
    IMAGE_EXTENSIONS,
)

# Command patterns
WEEK_PATTERN = re.compile(r'^/week\s+(\d+)\b', re.IGNORECASE)

# Week topics for student reference
WEEK_TOPICS = {
    1: "Finite Automata, NFA, Regular Operations",
    2: "Regular Expressions, Pumping Lemma",
    3: "Context-Free Grammars, Pushdown Automata",
    4: "Non-CFLs, CFL Pumping Lemma",
    5: "Turing Machines, Variants, Church-Turing",
    6: "Decidability, Undecidability",
    7: "Reducibility, Rice's Theorem",
    8: "Time Complexity, Class P",
    9: "Class NP, NP-completeness",
    10: "More NP-complete Problems",
    11: "Space Complexity, PSPACE",
    12: "L, NL, NL-completeness",
    13: "Hierarchy Theorems, Relativization, BPP",
}

# =============================================================================
# Model Configuration
# =============================================================================

QWEN_MODEL_SPECS = {
    # Qwen3 series - flagship models (Sept 2025)
    "qwen3-max": (256_000, 32_768),  # Best model: $1.20/$6.00 per 1M tokens
    "qwen3-max-latest": (256_000, 32_768),
    # Qwen API hosted models (qwen.ai/apiplatform)
    "qwen-max": (128_000, 8_192),  # Legacy, use qwen3-max instead
    "qwen-max-latest": (128_000, 8_192),
    "qwen-plus": (128_000, 8_192),  # Good balance: $0.40/$1.20 per 1M tokens
    "qwen-plus-latest": (128_000, 8_192),
    "qwen-turbo": (128_000, 8_192),  # Budget: $0.05/$0.20 per 1M tokens
    "qwen-turbo-latest": (128_000, 8_192),
    "qwen-long": (10_000_000, 8_192),  # Long context model
    # Qwen3 open-weight series
    "qwen3-235b-a22b": (22_000, 8_192),
    "qwen3-30b-a3b": (128_000, 8_192),
    "qwen3-32b": (128_000, 8_192),
    "qwen3-14b": (128_000, 8_192),
    "qwen3-8b": (128_000, 8_192),
    "qwen3-4b": (32_000, 8_192),
    "qwen3-1.7b": (32_000, 8_192),
    # Qwen2.5 series
    "qwen2.5-72b-instruct": (128_000, 8_192),
    "qwen2.5-32b-instruct": (128_000, 8_192),
    "qwen2.5-14b-instruct": (128_000, 8_192),
    "qwen2.5-7b-instruct": (128_000, 8_192),
    "qwen2.5-7b-instruct-1m": (1_000_000, 8_192),
    "qwen2.5-14b-instruct-1m": (1_000_000, 8_192),
    # Qwen-VL (vision-language)
    "qwen-vl-max": (32_000, 8_192),
    "qwen-vl-max-latest": (32_000, 8_192),
    "qwen-vl-plus": (32_000, 8_192),
    "qwen-vl-plus-latest": (32_000, 8_192),
    "qwen2.5-vl-72b-instruct": (128_000, 8_192),
    "qwen2.5-vl-7b-instruct": (128_000, 8_192),
    "qwen2.5-vl-3b-instruct": (128_000, 8_192),
    # QwQ (reasoning)
    "qwq-32b": (128_000, 16_384),
    "qwq-plus": (128_000, 16_384),
    "qwq-plus-latest": (128_000, 16_384),
}

DEFAULT_QWEN_CONTEXT = 128_000
DEFAULT_QWEN_OUTPUT = 8_192

# Vision-language models that support multimodal input
QWEN_VL_MODELS = {
    "qwen-vl-max", "qwen-vl-max-latest", 
    "qwen-vl-plus", "qwen-vl-plus-latest",
    "qwen2.5-vl-72b-instruct", "qwen2.5-vl-7b-instruct", "qwen2.5-vl-3b-instruct",
}


def get_qwen_model_specs(model: str) -> tuple[int, int]:
    """Get Qwen model specifications (context_window, max_output_tokens)."""
    model_lower = model.lower()
    
    if model_lower in QWEN_MODEL_SPECS:
        return QWEN_MODEL_SPECS[model_lower]
    
    # Fallback for unknown models based on name patterns
    if "qwen3-max" in model_lower:
        return (256_000, 32_768)
    elif "1m" in model_lower:
        return (1_000_000, 8_192)
    elif "qwq" in model_lower:
        return (128_000, 16_384)
    elif "vl" in model_lower:
        return (32_000, 8_192)
    elif "long" in model_lower:
        return (10_000_000, 8_192)
    
    logging.warning(f"Unknown Qwen model '{model}', using defaults")
    return (DEFAULT_QWEN_CONTEXT, DEFAULT_QWEN_OUTPUT)


def is_vision_model(model: str) -> bool:
    """Check if the model supports vision/multimodal input."""
    return model.lower() in QWEN_VL_MODELS or "vl" in model.lower()


# =============================================================================
# QwenChatBot Class
# =============================================================================

class QwenChatBot:
    """
    Qwen-powered chatbot using the Qwen-Agent framework.
    
    Features:
    - RAG via Qwen-Agent's built-in document retrieval
    - Week-based context for DM mode
    - Multimodal support with Qwen-VL models
    
    Commands:
    - /week N  - Set current week (e.g., /week 3)
    - /reset   - Clear conversation history and week selection
    """
    
    def __init__(
        self, 
        model: str, 
        api_key: str, 
        course_dir: str = None,
        file_patterns: list = None,
        max_output_tokens: int = None,
        log_qa: bool = False,
        file_downloader: Optional[Callable[[str], bytes | None]] = None,
        **kwargs  # Accept but ignore other backend-specific args
    ):
        """
        Initialize the Qwen chatbot with Qwen-Agent framework.
        
        Args:
            model: Qwen model name (e.g., qwen3-max, qwen-vl-max)
            api_key: DashScope API key
            course_dir: Path to course materials directory
            file_patterns: Patterns to filter course files
            max_output_tokens: Override for max output tokens
            log_qa: Whether to log each Q&A pair
            file_downloader: Callback to download files from Zulip
        """
        self.model = model
        self.api_key = api_key
        self.file_downloader = file_downloader
        
        # Set API key for DashScope
        os.environ["DASHSCOPE_API_KEY"] = api_key
        
        _, default_max_output = get_qwen_model_specs(model)
        self.max_output_tokens = max_output_tokens or default_max_output
        
        self.log_qa = log_qa
        self.file_patterns = file_patterns or []
        
        # Check if this is a vision model
        self.is_vision_model = is_vision_model(model)
        
        # LLM configuration for Qwen-Agent
        self.llm_cfg = {
            'model': model,
            'model_type': 'qwen_dashscope',
            'generate_cfg': {
                'max_output_tokens': self.max_output_tokens,
            }
        }
        
        # Load course materials (organized by week)
        self.course_materials = load_course_materials(course_dir)
        
        # Pre-compute context for each week
        self.week_contexts = self._prepare_week_contexts()
        
        # Legacy: full course context (for stream mode)
        self.course_context = self._prepare_course_context()
        
        # Per-user state
        self.conversations = {}  # user_id -> list of messages
        self.user_weeks = {}     # user_id -> current week number
        self.user_agents = {}    # user_id -> Assistant instance
        
        # Available weeks
        self.available_weeks = sorted([
            int(k.replace('week', '')) 
            for k in self.course_materials.keys() 
            if k.startswith('week')
        ])
        
        # Create base agent for stream mode
        self.stream_agent = self._create_agent(self._get_base_instructions())
        
        logging.info(f"QwenChatBot initialized with Qwen-Agent: model={model}")
        logging.info(f"Vision model: {self.is_vision_model}")
        logging.info(f"Available weeks: {self.available_weeks}")
        logging.info(f"Week contexts prepared: {list(self.week_contexts.keys())}")
    
    def _create_agent(self, system_prompt: str) -> Assistant:
        """Create a Qwen-Agent Assistant with the given system prompt."""
        return Assistant(
            llm=self.llm_cfg,
            system_message=system_prompt,
            name='Professor-GPT',
            description='Course assistant for DSAA3071 Theory of Computation',
        )
    
    def _prepare_week_contexts(self) -> dict:
        """Pre-compute context for each week for fast switching."""
        week_contexts = {}
        
        for week_key, files in self.course_materials.items():
            if not week_key.startswith('week'):
                continue
            
            week_num = int(week_key.replace('week', ''))
            week_parts = []
            
            for filename, content in sorted(files.items()):
                if self.file_patterns and not match_file_pattern(filename, self.file_patterns):
                    continue
                
                clean_content = extract_text_content(content)
                if clean_content.strip():
                    week_parts.append(f"### {filename}\n{clean_content}")
            
            if week_parts:
                context = f"## WEEK {week_num} MATERIALS\n\n" + "\n\n".join(week_parts)
                week_contexts[week_num] = context
                logging.info(f"Week {week_num}: {len(context):,} chars, {len(week_parts)} files")
        
        return week_contexts
    
    def _prepare_course_context(self) -> str:
        """Load all matching course materials into a single context string."""
        context_parts = []
        file_count = 0
        
        for week, files in sorted(self.course_materials.items()):
            week_parts = []
            for filename, content in sorted(files.items()):
                if self.file_patterns and not match_file_pattern(filename, self.file_patterns):
                    continue
                
                clean_content = extract_text_content(content)
                if clean_content.strip():
                    week_parts.append(f"### {filename}\n{clean_content}")
                    file_count += 1
            
            if week_parts:
                context_parts.append(f"## {week.upper()}\n\n" + "\n\n".join(week_parts))
        
        full_context = "\n\n---\n\n".join(context_parts)
        logging.info(f"Full context: {len(full_context):,} chars, {file_count} files")
        return full_context
    
    def _extract_files_from_message(self, message: str, user_display: str = None) -> tuple[str, list]:
        """Extract Zulip file URLs and download images for multimodal input.
        
        Returns:
            tuple: (cleaned_message, list of ContentItem for images)
        """
        if not self.file_downloader:
            return message, []
        
        file_urls = ZULIP_FILE_URL_PATTERN.findall(message)
        if not file_urls:
            return message, []
        
        image_items = []
        cleaned_message = message
        
        for file_url in file_urls:
            ext = os.path.splitext(file_url)[1].lower()
            filename = os.path.basename(file_url)
            
            is_image = ext in IMAGE_EXTENSIONS
            
            if not is_image:
                logging.info(f"[{user_display}] Skipping non-image file: {file_url}")
                continue
            
            if not self.is_vision_model:
                logging.info(f"[{user_display}] Skipping image (model doesn't support vision): {file_url}")
                cleaned_message = cleaned_message.replace(file_url, f"[image: {filename} - use qwen-vl-* model for vision]")
                continue
            
            try:
                file_bytes = self.file_downloader(file_url)
                if file_bytes:
                    b64_data = base64.b64encode(file_bytes).decode('utf-8')
                    mime_type = {
                        '.png': 'image/png',
                        '.jpg': 'image/jpeg',
                        '.jpeg': 'image/jpeg',
                        '.gif': 'image/gif',
                        '.webp': 'image/webp',
                    }.get(ext, 'image/png')
                    
                    # Qwen-Agent uses ContentItem for multimodal
                    image_items.append(ContentItem(
                        image=f"data:{mime_type};base64,{b64_data}"
                    ))
                    
                    logging.info(f"[{user_display}] Downloaded image: {file_url} ({len(file_bytes)} bytes)")
                    cleaned_message = cleaned_message.replace(file_url, f"[attached image: {filename}]")
                else:
                    logging.warning(f"[{user_display}] Failed to download file: {file_url}")
            except Exception as e:
                logging.error(f"[{user_display}] Error processing file {file_url}: {e}")
        
        return cleaned_message, image_items
    
    def _get_base_instructions(self) -> str:
        """Base instructions for the AI assistant."""
        return """You are a university professor teaching DSAA3071 Theory of Computation at HKUST(GZ).

=== ACCURACY IS YOUR TOP PRIORITY ===

CRITICAL RULES:
1. ONLY use information from the provided course materials when answering course-specific questions
2. If a question goes beyond the course materials, clearly state: "This goes beyond our course materials. Based on my general knowledge..."
3. If you're uncertain about something, SAY SO explicitly - do not guess or fabricate
4. For mathematical proofs and definitions, be PRECISE - errors in formal statements can mislead students
5. When in doubt, recommend the student verify with the textbook or ask in office hours

Your role is to:
- Help students understand concepts in automata theory, formal languages, and computability
- Explain DFA, NFA, regular expressions, context-free grammars, pushdown automata, and Turing machines
- Guide students through proofs and problem-solving techniques
- Be concise but thorough in explanations
- VERIFY your answers against the course materials before responding

As a professor, you occasionally:
- Draw connections to related topics (complexity theory, algorithms, programming languages, logic)
- Share insights from your professional knowledge to inspire deeper thinking
- Ask thought-provoking questions that encourage students to explore further
- Mention real-world applications or historical context when relevant

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
- "function$$f(x)$$is" - NO missing spaces for inline!
=== END MATH FORMATTING ==="""

    def _get_system_prompt(self, week_num: int = None) -> str:
        """Build system prompt with course materials embedded."""
        base = self._get_base_instructions()
        
        if week_num and week_num in self.week_contexts:
            context = self.week_contexts[week_num]
            return f"""{base}

=== WEEK {week_num} COURSE MATERIALS ===
You are currently helping a student with Week {week_num} content.
Use these materials to answer their questions accurately.

{context}

=== END COURSE MATERIALS ==="""
        else:
            return base
    
    def _quote_message(self, text: str) -> str:
        """Quote a message using Zulip markdown."""
        max_backticks = 0
        current_count = 0
        for char in text:
            if char == '`':
                current_count += 1
                max_backticks = max(max_backticks, current_count)
            else:
                current_count = 0
        
        fence = '`' * max(4, max_backticks + 1)
        return f"{fence}quote\n{text}\n{fence}"
    
    def _format_stream_response(
        self, 
        question: str, 
        reply: str, 
        sender_name: str = None, 
        sender_id: int = None, 
        message_url: str = None
    ) -> str:
        """Format stream response with user mention and quoted question."""
        quoted_question = self._quote_message(question)
        
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
            f"Model: {self.model}"
        )
    
    def _format_dm_response(self, reply: str, week_num: int = None) -> str:
        """Format DM response."""
        week_info = f"Week {week_num} | " if week_num else ""
        return (
            f"{reply}\n"
            f"------\n"
            f"{week_info}Model: {self.model}"
        )
    
    def _run_agent(self, agent: Assistant, messages: list) -> str:
        """Run the Qwen-Agent and collect the response."""
        response_text = ""
        
        for response in agent.run(messages=messages):
            # response is a list of message dicts
            if response:
                last_msg = response[-1]
                if isinstance(last_msg, dict) and 'content' in last_msg:
                    content = last_msg['content']
                    if isinstance(content, str):
                        response_text = content
                    elif isinstance(content, list):
                        # Handle list of ContentItems
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                text_parts.append(item['text'])
                            elif hasattr(item, 'text') and item.text:
                                text_parts.append(item.text)
                        response_text = ''.join(text_parts)
        
        return response_text or "I couldn't generate a response."
    
    def get_stream_response(
        self, 
        user_id: str, 
        prompt: str, 
        original_message: str = None,
        sender_name: str = None, 
        sender_id: int = None, 
        message_url: str = None
    ) -> str:
        """
        Handle stream message using Qwen-Agent.
        
        Stream mode uses base instructions without course materials context.
        """
        display_name = f"{sender_name} ({user_id})" if sender_name else user_id
        logging.info(f"Stream message from {display_name}")
        
        quote_text = original_message if original_message else prompt
        
        try:
            # Extract images if vision model
            cleaned_prompt, image_items = self._extract_files_from_message(prompt, display_name)
            
            # Build message content
            if image_items and self.is_vision_model:
                content = [ContentItem(text=cleaned_prompt)] + image_items
                messages = [{'role': 'user', 'content': content}]
                logging.info(f"[{display_name}] Stream: Using multimodal with {len(image_items)} image(s)")
            else:
                messages = [{'role': 'user', 'content': prompt}]
            
            # Run agent
            reply = self._run_agent(self.stream_agent, messages)
            
            if self.log_qa:
                logging.info(f"[Stream Q&A] User={user_id}\nQ: {prompt}\nA: {reply}")
            
            return self._format_stream_response(
                quote_text, reply, sender_name, sender_id, message_url
            )
            
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
        Handle DM message with week-based context using Qwen-Agent.
        
        Commands:
        - /week N  - Set current week (loads only that week's materials)
        - /reset   - Clear conversation and week selection
        """
        display_name = f"{user_name} ({user_id})" if user_name else user_id
        logging.info(f"DM message from {display_name}")
        
        prompt_stripped = prompt.strip()
        
        # Handle /reset command
        if RESET_PATTERN.match(prompt_stripped):
            self.clear_user_session(user_id)
            return "✅ Conversation cleared. Use `/week N` to select a week to study."
        
        # Handle /week N command
        week_match = WEEK_PATTERN.match(prompt_stripped)
        if week_match:
            week_num = int(week_match.group(1))
            return self._set_user_week(user_id, week_num)
        
        # Check if user has selected a week
        current_week = self.user_weeks.get(user_id)
        if current_week is None:
            weeks_table = "\n".join([
                f"- **Week {w}**: {WEEK_TOPICS.get(w, 'Topics TBD')}"
                for w in self.available_weeks
            ])
            return (
                f"👋 Welcome! Please select which week you want to study.\n\n"
                f"**Available weeks:**\n{weeks_table}\n\n"
                f"**Command:** `/week N` (e.g., `/week 3` for Week 3)\n\n"
                f"Once you select a week, I'll load the relevant course materials and help you learn!"
            )
        
        try:
            # Get or create agent for this user's week
            if user_id not in self.user_agents:
                system_prompt = self._get_system_prompt(current_week)
                self.user_agents[user_id] = self._create_agent(system_prompt)
                logging.info(f"Created agent for user {user_id} with week {current_week}")
            
            # Get or create conversation history
            if user_id not in self.conversations:
                self.conversations[user_id] = []
            
            history = self.conversations[user_id]
            
            # Extract images if vision model
            cleaned_prompt, image_items = self._extract_files_from_message(prompt, display_name)
            
            # Build message content
            if image_items and self.is_vision_model:
                content = [ContentItem(text=cleaned_prompt)] + image_items
                user_message = {'role': 'user', 'content': content}
                logging.info(f"[{display_name}] DM: Using multimodal with {len(image_items)} image(s)")
            else:
                user_message = {'role': 'user', 'content': prompt}
            
            history.append(user_message)
            
            # Run agent with full history
            agent = self.user_agents[user_id]
            reply = self._run_agent(agent, history)
            
            # Add assistant response to history
            history.append({'role': 'assistant', 'content': reply})
            
            # Trim history if too long (keep last 10 messages)
            max_history = 10
            if len(history) > max_history:
                self.conversations[user_id] = history[-max_history:]
                logging.info(f"Trimmed conversation history for {user_id}")
            
            if self.log_qa:
                logging.info(f"[DM Q&A] User={display_name} Week={current_week}\nQ: {prompt}\nA: {reply}")
            
            return self._format_dm_response(reply, current_week)
            
        except Exception as e:
            logging.error(f"DM response error: {e}")
            return f"Error: {str(e)}"
    
    def _set_user_week(self, user_id: str, week_num: int) -> str:
        """Set the current week for a user and create a new agent."""
        if week_num not in self.available_weeks:
            weeks_list = ", ".join(str(w) for w in self.available_weeks)
            return f"❌ Week {week_num} not found. Available weeks: {weeks_list}"
        
        old_week = self.user_weeks.get(user_id)
        self.user_weeks[user_id] = week_num
        
        # Clear conversation and agent when switching weeks
        if old_week != week_num:
            if user_id in self.conversations:
                del self.conversations[user_id]
            if user_id in self.user_agents:
                del self.user_agents[user_id]
            logging.info(f"Cleared session for {user_id} due to week change")
        
        # Create new agent with week context
        system_prompt = self._get_system_prompt(week_num)
        self.user_agents[user_id] = self._create_agent(system_prompt)
        
        context_size = len(self.week_contexts.get(week_num, ""))
        topic = WEEK_TOPICS.get(week_num, "Course materials")
        logging.info(f"User {user_id} set to week {week_num} ({context_size:,} chars)")
        
        return (
            f"✅ **Week {week_num}: {topic}**\n\n"
            f"I've loaded the Week {week_num} course materials ({context_size:,} characters).\n"
            f"Ask me anything about this week's topics!\n\n"
            f"_Tip: Use `/reset` to clear our conversation, or `/week N` to switch weeks._"
        )
    
    def clear_user_session(self, user_id: str):
        """Clear a user's conversation history, week selection, and agent."""
        if user_id in self.conversations:
            del self.conversations[user_id]
        if user_id in self.user_weeks:
            del self.user_weeks[user_id]
        if user_id in self.user_agents:
            del self.user_agents[user_id]
        logging.info(f"Cleared session for user {user_id}")
