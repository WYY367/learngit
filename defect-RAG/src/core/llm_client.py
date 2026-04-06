"""LLM client using OpenAI Compatible API."""
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """LLM client for OpenAI Compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        default_params: Optional[Dict[str, Any]] = None,
        verify_ssl: bool = True
    ):
        """Initialize LLM client.

        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            default_params: Default generation parameters
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.default_params = default_params or {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        self.verify_ssl = verify_ssl

        # Initialize client with SSL configuration
        if verify_ssl:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
        else:
            # Disable SSL verification
            http_client = httpx.Client(verify=False)
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=http_client
            )
            logger.warning("SSL verification is disabled. This is not recommended for production.")

        logger.info(f"Initialized LLM client with model: {model}")

    def invoke(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Invoke LLM with messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override default parameters

        Returns:
            Generated text
        """
        # Merge parameters
        params = {**self.default_params, **kwargs}

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                frequency_penalty=params.get("frequency_penalty"),
                presence_penalty=params.get("presence_penalty")
            )

            content = response.choices[0].message.content
            logger.info(f"LLM invocation successful, tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
            return content

        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

    def invoke_with_system(
        self,
        user_message: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Convenience method with system message.

        Args:
            user_message: User input
            system_message: System prompt
            **kwargs: Override default parameters

        Returns:
            Generated text
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})

        return self.invoke(messages, **kwargs)

    def update_default_params(self, **params) -> None:
        """Update default parameters.

        Args:
            **params: Parameters to update
        """
        self.default_params.update(params)
        logger.info(f"Updated default params: {params}")

    def get_default_params(self) -> Dict[str, Any]:
        """Get current default parameters.

        Returns:
            Dictionary of default parameters
        """
        return self.default_params.copy()
