# claude.py
"""
Claude-powered course assistant for DSAA3071 Theory of Computation.

Two modes:
- STREAM (public): No course materials, answers from general knowledge (~1K tokens)
- DM (private): Week-based context with prompt caching (~6K tokens)

DM Flow:
1. Student DMs the bot
2. Bot prompts: "Use /week N to select a week"
3. Student types: /week 3
4. Bot loads Week 3 materials with prompt caching
5. Student asks questions, follow-ups use cached context (~90% savings)

Commands (DM only):
- /week N  - Select which week to study (e.g., /week 3)
- /reset   - Clear conversation and week selection
"""

import anthropic
import logging
import os
import re

from chatgpt import (
    load_course_materials, 
    match_file_pattern, 
    extract_text_content,
    RESET_PATTERN
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

CLAUDE_MODEL_SPECS = {
    # Claude 4.5 series (latest)
    "claude-sonnet-4-5-20250929": (200_000, 16_384),  # Recommended
    "claude-opus-4-5-20251101": (200_000, 32_000),
    "claude-haiku-4-5-20251001": (200_000, 8_192),
    # Claude 4.1 series
    "claude-opus-4-1-20250805": (200_000, 32_000),
    # Claude 4 series
    "claude-sonnet-4-20250514": (200_000, 16_384),
    "claude-opus-4-20250514": (200_000, 32_000),
    # Claude 3.5 series
    "claude-3-5-haiku-20241022": (200_000, 8_192),
    # Claude 3 series
    "claude-3-haiku-20240307": (200_000, 4_096),
}

DEFAULT_CLAUDE_CONTEXT = 200_000
DEFAULT_CLAUDE_OUTPUT = 8_192


def get_claude_model_specs(model: str) -> tuple[int, int]:
    """Get Claude model specifications (context_window, max_output_tokens)."""
    if model in CLAUDE_MODEL_SPECS:
        return CLAUDE_MODEL_SPECS[model]
    
    # Fallback for unknown models based on tier
    model_lower = model.lower()
    if "opus" in model_lower:
        return (200_000, 32_000)
    elif "sonnet" in model_lower:
        return (200_000, 16_384)
    elif "haiku" in model_lower:
        return (200_000, 8_192)
    elif "4.5" in model_lower or "4-5" in model_lower:
        return (200_000, 16_384)  # Claude 4.5 default
    
    logging.warning(f"Unknown Claude model '{model}', using defaults")
    return (DEFAULT_CLAUDE_CONTEXT, DEFAULT_CLAUDE_OUTPUT)


# =============================================================================
# ClaudeChatBot Class
# =============================================================================

class ClaudeChatBot:
    """
    Claude-powered chatbot with week-based context loading and prompt caching.
    
    Students select which week to study with /week N command.
    Only that week's materials are loaded (~6K tokens instead of 80K).
    Prompt caching reduces costs by ~90% for follow-up questions.
    
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
        **kwargs  # Accept but ignore OpenAI-specific args like vector_store_id
    ):
        """
        Initialize the Claude chatbot.
        
        Args:
            model: Claude model name (e.g., claude-sonnet-4-5-20250514)
            api_key: Anthropic API key
            course_dir: Path to course materials directory
            file_patterns: Patterns to filter course files
            max_output_tokens: Override for max output tokens
            log_qa: Whether to log each Q&A pair
        """
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key)
        
        _, default_max_output = get_claude_model_specs(model)
        self.max_output_tokens = max_output_tokens or default_max_output
        
        self.log_qa = log_qa
        self.file_patterns = file_patterns or []
        
        # Load course materials (organized by week)
        self.course_materials = load_course_materials(course_dir)
        
        # Pre-compute context for each week (for fast switching)
        self.week_contexts = self._prepare_week_contexts()
        
        # Legacy: full course context (for stream mode fallback)
        self.course_context = self._prepare_course_context()
        
        # Per-user state
        self.conversations = {}  # user_id -> list of messages
        self.user_weeks = {}     # user_id -> current week number
        
        # Available weeks
        self.available_weeks = sorted([
            int(k.replace('week', '')) 
            for k in self.course_materials.keys() 
            if k.startswith('week')
        ])
        
        logging.info(f"ClaudeChatBot initialized: model={model}")
        logging.info(f"Available weeks: {self.available_weeks}")
        logging.info(f"Week contexts prepared: {list(self.week_contexts.keys())}")
    
    def _prepare_week_contexts(self) -> dict:
        """Pre-compute context for each week for fast switching."""
        week_contexts = {}
        
        for week_key, files in self.course_materials.items():
            if not week_key.startswith('week'):
                continue
            
            week_num = int(week_key.replace('week', ''))
            week_parts = []
            
            for filename, content in sorted(files.items()):
                # Apply file pattern filter
                if self.file_patterns and not match_file_pattern(filename, self.file_patterns):
                    continue
                
                # Extract readable content
                clean_content = extract_text_content(content)
                if clean_content.strip():
                    week_parts.append(f"### {filename}\n{clean_content}")
            
            if week_parts:
                context = f"## WEEK {week_num} MATERIALS\n\n" + "\n\n".join(week_parts)
                week_contexts[week_num] = context
                logging.info(f"Week {week_num}: {len(context):,} chars, {len(week_parts)} files")
        
        return week_contexts
    
    def _prepare_course_context(self) -> str:
        """Load all matching course materials into a single context string (fallback)."""
        context_parts = []
        file_count = 0
        
        for week, files in sorted(self.course_materials.items()):
            week_parts = []
            for filename, content in sorted(files.items()):
                # Apply file pattern filter
                if self.file_patterns and not match_file_pattern(filename, self.file_patterns):
                    continue
                
                # Extract readable content
                clean_content = extract_text_content(content)
                if clean_content.strip():
                    week_parts.append(f"### {filename}\n{clean_content}")
                    file_count += 1
            
            if week_parts:
                context_parts.append(f"## {week.upper()}\n\n" + "\n\n".join(week_parts))
        
        full_context = "\n\n---\n\n".join(context_parts)
        logging.info(f"Full context: {len(full_context):,} chars, {file_count} files")
        return full_context
    
    def _get_week_context(self, week_num: int) -> str:
        """Get context for a specific week."""
        if week_num in self.week_contexts:
            return self.week_contexts[week_num]
        return ""
    
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

When a student asks about something NOT covered in the course materials:
- First check if it's a general theory of computation topic (answer from knowledge, but note it's beyond course scope)
- For current events, recent research, or things that require up-to-date information, acknowledge you may not have the latest information

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
            # Week-specific context (efficient)
            context = self.week_contexts[week_num]
            return f"""{base}

=== WEEK {week_num} COURSE MATERIALS ===
You are currently helping a student with Week {week_num} content.
Use these materials to answer their questions accurately.

{context}

=== END COURSE MATERIALS ==="""
        elif self.course_context:
            # Full course context (fallback for stream mode)
            return f"""{base}

=== COURSE MATERIALS ===
Below are the course materials. Use these to answer student questions accurately.

{self.course_context}

=== END COURSE MATERIALS ==="""
        else:
            return base
    
    def _get_system_messages(self, week_num: int = None) -> list:
        """Build system messages with prompt caching enabled."""
        base = self._get_base_instructions()
        
        if week_num and week_num in self.week_contexts:
            context = self.week_contexts[week_num]
            context_text = f"""=== WEEK {week_num} COURSE MATERIALS ===
You are currently helping a student with Week {week_num} content.
Use these materials to answer their questions accurately.

{context}

=== END COURSE MATERIALS ==="""
        elif self.course_context:
            context_text = f"""=== COURSE MATERIALS ===
Below are the course materials. Use these to answer student questions accurately.

{self.course_context}

=== END COURSE MATERIALS ==="""
        else:
            context_text = None
        
        # Build system messages with caching
        messages = [{"type": "text", "text": base}]
        
        if context_text:
            messages.append({
                "type": "text",
                "text": context_text,
                "cache_control": {"type": "ephemeral"}  # Enable prompt caching
            })
        
        return messages
    
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
        input_tokens: int,
        output_tokens: int,
        sender_name: str = None, 
        sender_id: int = None, 
        message_url: str = None
    ) -> str:
        """Format stream response with user mention, quoted question, and usage info."""
        quoted_question = self._quote_message(question)
        
        if sender_name and sender_id and message_url:
            mention = f"@_**{sender_name}|{sender_id}** [said]({message_url}):\n"
        elif sender_name and sender_id:
            mention = f"@_**{sender_name}|{sender_id}**:\n"
        else:
            mention = ""
        
        total_tokens = input_tokens + output_tokens
        return (
            f"{mention}{quoted_question}\n\n"
            f"{reply}\n"
            f"------\n"
            f"Tokens: {input_tokens:,} (input) + {output_tokens:,} (output) "
            f"= {total_tokens:,}"
        )
    
    def _format_dm_response(self, reply: str, input_tokens: int, output_tokens: int) -> str:
        """Format DM response with just reply and usage info."""
        total_tokens = input_tokens + output_tokens
        return (
            f"{reply}\n"
            f"------\n"
            f"Tokens: {input_tokens:,} (input) + {output_tokens:,} (output) "
            f"= {total_tokens:,}"
        )
    
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
        Handle stream message: NO course materials loaded.
        
        Stream mode uses only the base instructions (no textbook context).
        This keeps costs low for public channel questions.
        For course-specific questions, students should DM with /week N.
        """
        logging.info(f"Stream message from {user_id} (no course context)")
        
        quote_text = original_message if original_message else prompt
        
        try:
            # Use only base instructions - NO course materials
            system_prompt = self._get_base_instructions() + """

NOTE: You do NOT have access to course materials in this mode.
For general theory of computation questions, answer from your knowledge.
For course-specific questions (homework, specific lecture content), 
suggest the student DM you with /week N to load the relevant materials."""
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}],
            )
            
            reply = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            if self.log_qa:
                logging.info(f"[Stream Q&A] User={user_id}\nQ: {prompt}\nA: {reply}")
            
            return self._format_stream_response(
                quote_text, reply, input_tokens, output_tokens,
                sender_name, sender_id, message_url
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
    
    def get_dm_response(self, user_id: str, prompt: str) -> str:
        """
        Handle DM message with week-based context and prompt caching.
        
        Commands:
        - /week N  - Set current week (loads only that week's materials)
        - /reset   - Clear conversation and week selection
        """
        logging.info(f"DM message from {user_id}")
        
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
            # Prompt user to select a week with topics
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
            # Get or create conversation history
            if user_id not in self.conversations:
                self.conversations[user_id] = []
                logging.info(f"Created new conversation for user {user_id}")
            
            history = self.conversations[user_id]
            history.append({"role": "user", "content": prompt})
            
            # Use system messages with caching for efficiency
            system_messages = self._get_system_messages(current_week)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_output_tokens,
                system=system_messages,
                messages=history,
            )
            
            reply = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            # Get cache info if available
            cache_read = getattr(response.usage, 'cache_read_input_tokens', 0)
            cache_write = getattr(response.usage, 'cache_creation_input_tokens', 0)
            
            # Add assistant response to history
            history.append({"role": "assistant", "content": reply})
            
            # Trim history if it gets too long (keep last 5 exchanges for cost efficiency)
            max_history = 10  # 5 user + 5 assistant messages (~57% cost savings)
            if len(history) > max_history:
                self.conversations[user_id] = history[-max_history:]
                logging.info(f"Trimmed conversation history for {user_id}")
            
            if self.log_qa:
                logging.info(f"[DM Q&A] User={user_id} Week={current_week}\nQ: {prompt}\nA: {reply}")
            
            return self._format_dm_response_with_cache(
                reply, input_tokens, output_tokens, 
                cache_read, cache_write, current_week
            )
            
        except Exception as e:
            logging.error(f"DM response error: {e}")
            return f"Error: {str(e)}"
    
    def _set_user_week(self, user_id: str, week_num: int) -> str:
        """Set the current week for a user."""
        if week_num not in self.available_weeks:
            weeks_list = ", ".join(str(w) for w in self.available_weeks)
            return f"❌ Week {week_num} not found. Available weeks: {weeks_list}"
        
        old_week = self.user_weeks.get(user_id)
        self.user_weeks[user_id] = week_num
        
        # Clear conversation when switching weeks
        if old_week != week_num and user_id in self.conversations:
            del self.conversations[user_id]
            logging.info(f"Cleared conversation for {user_id} due to week change")
        
        context_size = len(self.week_contexts.get(week_num, ""))
        topic = WEEK_TOPICS.get(week_num, "Course materials")
        logging.info(f"User {user_id} set to week {week_num} ({context_size:,} chars)")
        
        return (
            f"✅ **Week {week_num}: {topic}**\n\n"
            f"I've loaded the Week {week_num} course materials ({context_size:,} characters).\n"
            f"Ask me anything about this week's topics!\n\n"
            f"_Tip: Use `/reset` to clear our conversation, or `/week N` to switch weeks._"
        )
    
    def _format_dm_response_with_cache(
        self, reply: str, input_tokens: int, output_tokens: int,
        cache_read: int, cache_write: int, week_num: int
    ) -> str:
        """Format DM response with cache info."""
        total_tokens = input_tokens + output_tokens
        
        # Calculate savings if cache was used
        cache_info = ""
        if cache_read > 0:
            # Cache read tokens cost 10% of normal
            savings = cache_read * 0.9  # 90% savings on cached tokens
            cache_info = f" | 📦 Cache: {cache_read:,} tokens (saved ~90%)"
        elif cache_write > 0:
            cache_info = f" | 📝 Cached: {cache_write:,} tokens"
        
        return (
            f"{reply}\n"
            f"------\n"
            f"Week {week_num} | Tokens: {input_tokens:,} + {output_tokens:,} = {total_tokens:,}{cache_info}"
        )
    
    def clear_user_session(self, user_id: str):
        """Clear a user's conversation history and week selection."""
        if user_id in self.conversations:
            del self.conversations[user_id]
        if user_id in self.user_weeks:
            del self.user_weeks[user_id]
        logging.info(f"Cleared session for user {user_id}")
