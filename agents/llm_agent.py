"""
LLM-backed agent implementation.

Agents that use language models (OpenAI, Anthropic, Google) to make bargaining decisions.
"""

import os
import re
from typing import Any, Dict, Optional, Literal
from .base import Agent
from .personas import get_system_prompt, format_observation_prompt, build_agent_prompt


class LLMAgent(Agent):
    """
    Agent that uses an LLM to make decisions in bargaining games.

    Supports:
    - OpenAI (GPT-3.5, GPT-4, GPT-4o, o1-preview)
    - Anthropic (Claude 3.5 Sonnet, Claude 4.5)
    - Google (Gemini 1.5 Pro)
    
    By default, agents receive only the neutral MASTER_SYSTEM_PROMPT with P1/P2 role.
    Persona injection is opt-in via use_persona=True for experimental conditions.
    """

    def __init__(
        self,
        agent_id: str,
        persona_type: str = "default",
        model: str = "gpt-3.5-turbo",
        provider: Literal["openai", "anthropic", "google", "xai"] = "openai",
        temperature: float = 0.7,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        use_persona: bool = False,
    ):
        """
        Initialize LLM agent.

        Args:
            agent_id: Unique identifier for this agent
            persona_type: Type of persona (only used if use_persona=True)
            model: Model name (e.g., 'gpt-5.1', 'claude-opus-4-5-20251101', 'gemini-3-pro-preview')
            provider: API provider ('openai', 'anthropic', 'google', or 'xai')
            temperature: Sampling temperature (0 = deterministic, 1 = creative)
            max_retries: Maximum retries for API calls
            api_key: API key (if None, reads from environment)
            reasoning_effort: For OpenAI reasoning models, set to 'low', 'medium', or 'high'
            use_persona: If True, inject persona instructions (experimental mode)
        """
        super().__init__(agent_id, persona_type)
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self.use_persona = use_persona
        
        # Store API call history for debugging/logging
        self.api_call_history = []

        # Get system prompt (neutral by default, persona only if use_persona=True)
        self.system_prompt = get_system_prompt(
            agent_id=agent_id,
            use_persona=use_persona,
            persona_type=persona_type if use_persona else None,
        )

        # Initialize API client
        self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]):
        """Initialize the appropriate API client."""
        if self.provider == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

            # Check multiple env var names for compatibility
            key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
            if not key:
                raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or OPENAI_KEY, or pass api_key.")

            self.client = openai.OpenAI(api_key=key)

        elif self.provider == "xai":
            try:
                import openai
            except ImportError:
                raise ImportError("openai package required for xAI. Install with: pip install openai")
            
            key = api_key or os.getenv("XAI_API_KEY")
            if not key:
                raise ValueError("xAI API key required. Set XAI_API_KEY.")
                
            self.client = openai.OpenAI(
                api_key=key,
                base_url="https://api.x.ai/v1"
            )

        elif self.provider == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError("anthropic package required. Install with: pip install anthropic")

            # Check multiple env var names for compatibility
            key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY")
            if not key:
                raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or ANTHROPIC_KEY, or pass api_key.")

            self.client = anthropic.Anthropic(api_key=key)

        elif self.provider == "google":
            try:
                from google import genai
            except ImportError:
                raise ImportError("google-genai package required. Install with: pip install google-genai")

            # Check multiple env var names for compatibility
            key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not key:
                raise ValueError("Google API key required. Set GOOGLE_API_KEY or GEMINI_API_KEY, or pass api_key.")

            self.client = genai.Client(api_key=key)
            self.google_model = self.model  # Store model name for generate_content call

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def choose_action(self, observation: Dict[str, Any]) -> Any:
        """
        Choose an action based on observation using LLM.

        Args:
            observation: Current game state observation

        Returns:
            Action (type depends on game and phase)
        """
        # Format observation into prompt
        user_prompt = format_observation_prompt(observation)

        # Get LLM response
        response_text = self._call_llm(user_prompt)

        # Parse response into action
        action = self._parse_response(response_text, observation)

        return action

    def generate_text(self, prompt: str) -> str:
        """
        Generate free-form text using the LLM.
        Useful for bulletin board posts or other non-game actions.
        """
        return self._call_llm(prompt)

    def _call_llm(self, user_prompt: str) -> str:
        """
        Call the LLM API with retry logic.

        Args:
            user_prompt: User message to send

        Returns:
            LLM response text
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    return self._call_openai(user_prompt)
                elif self.provider == "xai":
                    return self._call_openai(user_prompt)  # Uses same client interface
                elif self.provider == "anthropic":
                    return self._call_anthropic(user_prompt)
                elif self.provider == "google":
                    return self._call_google(user_prompt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM API call failed after {self.max_retries} attempts: {e}")
                # Retry
                continue

    def _call_openai(self, user_prompt: str) -> str:
        """Call OpenAI API.
        
        Notes:
        - Automatic caching for prompts >1024 tokens with identical prefix
        - Check response.usage.prompt_tokens_details.cached_tokens for cache hits
        - GPT-5.x models use max_completion_tokens instead of max_tokens
        - reasoning_effort param for GPT-5.x/o1/o3 models
        """
        # Build request parameters
        # GPT-5.x models use max_completion_tokens instead of max_tokens
        is_reasoning_model = self.model.startswith(('gpt-5', 'o1', 'o3'))
        
        params = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
        }
        
        # Add reasoning effort for o1/o3/gpt-5.x models (top-level parameter)
        if self.reasoning_effort and is_reasoning_model:
            params["reasoning_effort"] = self.reasoning_effort
        
        response = self.client.chat.completions.create(**params)
        response_text = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
        
        # Extract reasoning content from GPT-5.x/o1/o3 models
        reasoning_content = None
        message = response.choices[0].message
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            reasoning_content = message.reasoning_content
        # Some models put reasoning in a separate field
        elif hasattr(message, 'reasoning') and message.reasoning:
            reasoning_content = message.reasoning
        
        # Log full request and response
        self.api_call_history.append({
            "provider": "openai",
            "request": params,
            "response": {
                "id": response.id,
                "model": response.model,
                "content": response_text,
                "reasoning_content": reasoning_content,
                "usage": response.usage.model_dump() if response.usage else None,
                "finish_reason": response.choices[0].finish_reason,
            }
        })
        
        return response_text

    def _call_anthropic(self, user_prompt: str) -> str:
        """Call Anthropic API with prompt caching and extended thinking enabled."""
        # cache_control: ephemeral caches for 5 min (default TTL)
        # Cache reads cost 10% of base input price; writes cost 125%
        # Min tokens: 1024 (Sonnet/Opus 4), 4096 (Opus 4.5), 2048 (Haiku)
        
        # Extended thinking is enabled for Claude 4+ models
        # Returns summarized thinking in "thinking" content blocks
        request_params = {
            "model": self.model,
            "max_tokens": 16000,  # Must be > budget_tokens for extended thinking
            "temperature": 1,  # Extended thinking requires temperature=1
            "thinking": {
                "type": "enabled",
                "budget_tokens": 10000,  # Allow up to 10k tokens for reasoning
            },
            "system": [
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        
        response = self.client.messages.create(**request_params)
        
        # Extract thinking and text content from response
        thinking_content = None
        response_text = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_content = block.thinking
            elif block.type == "text":
                response_text = block.text.strip()
        
        # Log full request and response
        self.api_call_history.append({
            "provider": "anthropic",
            "request": request_params,
            "response": {
                "id": response.id,
                "model": response.model,
                "content": response_text,
                "thinking_content": thinking_content,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(response.usage, 'cache_creation_input_tokens', None),
                    "cache_read_input_tokens": getattr(response.usage, 'cache_read_input_tokens', None),
                },
                "stop_reason": response.stop_reason,
            }
        })
        
        return response_text

    def _call_google(self, user_prompt: str) -> str:
        """Call Google Gemini API using Gen AI SDK.
        
        Notes:
        - Gemini 3 recommends temperature=1.0 for optimal reasoning performance
        - Implicit caching is automatic for prompts > 2048 tokens with same prefix
        - No code needed for caching - Google handles it automatically
        """
        from google.genai import types
        
        # Gemini 3 Pro uses the new genai client
        # System instruction + user content
        full_prompt = f"{self.system_prompt}\n\n{user_prompt}"
        
        # Enable thought summaries to capture reasoning
        # No max_output_tokens - let model complete naturally
        # Note: For Gemini 3, temperature should be 1.0 (set via ModelConfig)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            thinking_config=types.ThinkingConfig(
                include_thoughts=True  # Get thought summaries!
            )
        )
        
        response = self.client.models.generate_content(
            model=self.google_model,
            contents=full_prompt,
            config=config,
        )
        
        # Handle potential None text
        response_text = response.text.strip() if response.text else ""
        
        # Extract thinking/reasoning content from Gemini thinking models
        # When include_thoughts=True, parts with thought=True contain the reasoning
        thinking_content = None
        if hasattr(response, 'candidates') and response.candidates:
            thoughts = []
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    parts = getattr(candidate.content, 'parts', []) or []
                    for part in parts:
                        # Parts with thought=True are the thinking summaries
                        if hasattr(part, 'thought') and part.thought:
                            if hasattr(part, 'text') and part.text:
                                thoughts.append(part.text)
            if thoughts:
                thinking_content = "\n".join(thoughts)
        
        # Log full request and response
        self.api_call_history.append({
            "provider": "google",
            "request": {
                "model": self.google_model,
                "contents": full_prompt,
                "config": {
                    "temperature": self.temperature,
                    "include_thoughts": True,
                },
            },
            "response": {
                "content": response_text,
                "thinking_content": thinking_content,
                "usage_metadata": {
                    "prompt_token_count": getattr(response.usage_metadata, 'prompt_token_count', None),
                    "candidates_token_count": getattr(response.usage_metadata, 'candidates_token_count', None),
                    "thoughts_token_count": getattr(response.usage_metadata, 'thoughts_token_count', None),
                    "total_token_count": getattr(response.usage_metadata, 'total_token_count', None),
                } if hasattr(response, 'usage_metadata') and response.usage_metadata else None,
            }
        })
        
        return response_text

    def _parse_response(self, response_text: str, observation: Dict[str, Any]) -> Any:
        """
        Parse LLM response into an action.

        Args:
            response_text: Raw LLM response
            observation: Current observation (for context)

        Returns:
            Parsed action
        """
        game_type = observation.get("game_type", "unknown")
        
        # Extract content inside [[...]]
        match = re.search(r"\[\[(.*?)\]\]", response_text, re.DOTALL)
        if match:
            action_text = match.group(1).strip()
        else:
            # Fallback: use the whole text if no brackets found (for backward compatibility or error recovery)
            action_text = response_text.strip()
        
        # For negotiation arena, return the full message as-is (with extracted offer if present)
        if game_type == "negotiation_arena":
            return self._parse_negotiation_action(action_text)
            
        response_lower = action_text.lower()

        # Check if it's an accept/reject decision
        if "accept" in response_lower or "reject" in response_lower:
            # For ultimatum and rubinstein response phases
            if "accept" in response_lower:
                return True
            elif "reject" in response_lower:
                return False
            else:
                # Ambiguous - default to accept
                return True

        # Otherwise, try to extract a number (for offers/demands)
        numbers = self._extract_numbers(action_text)

        if not numbers:
            # If we failed to extract from the bracketed text, try the whole text as a last resort
            numbers = self._extract_numbers(response_text)
            
        if not numbers:
            # Fallback: try to extract any numeric-like string
            raise ValueError(f"Could not parse numeric action from: {response_text}")

        # Return the first number found
        return numbers[0]
    
    def _parse_negotiation_action(self, action_text: str) -> Dict[str, Any]:
        """
        Parse a negotiation arena action from the agent's message.
        
        Returns a dict with: {"move": MoveType, "message": str, "offer": float|None, "deal": tuple|None}
        """
        action_lower = action_text.lower()
        
        # Check for $$DEAL: X / Y$$ marker (signals final agreement)
        deal_match = re.search(r"\$\$DEAL:\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*\$\$", action_text)
        if deal_match:
            my_amount = float(deal_match.group(1))
            their_amount = float(deal_match.group(2))
            return {
                "move": "accept", 
                "message": action_text, 
                "offer": their_amount,  # The offer being accepted (what they get)
                "deal": (my_amount, their_amount),  # (my_share, their_share)
                "deal_reached": True
            }
        
        # Detect move type
        if "walk away" in action_lower or "walk_away" in action_lower:
            return {"move": "walk_away", "message": action_text, "offer": None}
        elif "accept" in action_lower:
            # Accept without the $$DEAL$$ marker - try to extract numbers anyway
            numbers = self._extract_numbers(action_text)
            return {"move": "accept", "message": action_text, "offer": numbers[0] if numbers else None}
        elif "reject" in action_lower:
            # Check if there's also a counter-offer
            numbers = self._extract_numbers(action_text)
            if numbers:
                return {"move": "counter", "message": action_text, "offer": numbers[0]}
            return {"move": "reject", "message": action_text, "offer": None}
        else:
            # Look for a proposal/offer
            numbers = self._extract_numbers(action_text)
            if numbers:
                return {"move": "propose", "message": action_text, "offer": numbers[0]}
            # Just a message
            return {"move": "message", "message": action_text, "offer": None}

    def _extract_numbers(self, text: str) -> list:
        """
        Extract numeric values from text.

        Args:
            text: Text to parse

        Returns:
            List of numbers found
        """
        # Match integers and floats
        pattern = r"[-+]?\d*\.?\d+"
        matches = re.findall(pattern, text)
        return [float(m) for m in matches if m]


class MockAgent(Agent):
    """
    Mock agent for testing that returns pre-specified actions.

    Does not make API calls - useful for unit tests and development.
    """

    def __init__(
        self,
        agent_id: str,
        persona_type: str,
        strategy: Literal["equal_split", "greedy", "generous", "random"] = "equal_split",
        fixed_action: Optional[Any] = None,
    ):
        """
        Initialize mock agent.

        Args:
            agent_id: Unique identifier
            persona_type: Type of persona (for consistency with real agents)
            strategy: Behavioral strategy for choosing actions
            fixed_action: If provided, always return this action
        """
        super().__init__(agent_id, persona_type)
        self.strategy = strategy
        self.fixed_action = fixed_action

    def choose_action(self, observation: Dict[str, Any]) -> Any:
        """Choose action based on strategy."""
        if self.fixed_action is not None:
            return self.fixed_action

        game_type = observation.get("game_type")
        pie_size = observation.get("pie_size", 10)

        # Handle different game types
        if game_type == "nash_demand":
            return self._choose_nash_demand(pie_size)

        elif game_type == "ultimatum":
            role = observation.get("role")
            if role == "proposer":
                return self._choose_ultimatum_offer(pie_size)
            else:
                offer = observation.get("offer")
                return self._choose_ultimatum_response(offer, pie_size)

        elif game_type == "rubinstein":
            waiting_for = observation.get("waiting_for")
            if waiting_for == "offer":
                return self._choose_rubinstein_offer(pie_size)
            else:
                offer = observation.get("opponent_offer")
                return self._choose_rubinstein_response(offer, pie_size)

        # Fallback
        return pie_size / 2

    def _choose_nash_demand(self, pie_size: float) -> float:
        """Choose Nash Demand Game action."""
        if self.strategy == "equal_split":
            return pie_size / 2
        elif self.strategy == "greedy":
            return pie_size * 0.7
        elif self.strategy == "generous":
            return pie_size * 0.3
        elif self.strategy == "random":
            import random
            return random.uniform(0, pie_size)
        return pie_size / 2

    def _choose_ultimatum_offer(self, pie_size: float) -> float:
        """Choose Ultimatum Game offer."""
        if self.strategy == "equal_split":
            return pie_size / 2
        elif self.strategy == "greedy":
            return pie_size * 0.8
        elif self.strategy == "generous":
            return pie_size * 0.3
        elif self.strategy == "random":
            import random
            return random.uniform(0, pie_size)
        return pie_size / 2

    def _choose_ultimatum_response(self, offer: float, pie_size: float) -> bool:
        """Choose Ultimatum Game response."""
        you_receive = pie_size - offer

        if self.strategy == "equal_split":
            # Accept if you get >= 40%
            return you_receive >= pie_size * 0.4
        elif self.strategy == "greedy":
            # Accept only if you get >= 50%
            return you_receive >= pie_size * 0.5
        elif self.strategy == "generous":
            # Accept anything > 0
            return you_receive > 0
        elif self.strategy == "random":
            import random
            return random.choice([True, False])
        return you_receive >= pie_size * 0.4

    def _choose_rubinstein_offer(self, pie_size: float) -> float:
        """Choose Rubinstein offer."""
        # Similar to Nash Demand
        return self._choose_nash_demand(pie_size)

    def _choose_rubinstein_response(self, offer: float, pie_size: float) -> bool:
        """Choose Rubinstein response."""
        you_receive = pie_size - offer

        if self.strategy == "equal_split":
            return you_receive >= pie_size * 0.4
        elif self.strategy == "greedy":
            return you_receive >= pie_size * 0.5
        elif self.strategy == "generous":
            return you_receive > 0
        elif self.strategy == "random":
            import random
            return random.choice([True, False])
        return you_receive >= pie_size * 0.4
