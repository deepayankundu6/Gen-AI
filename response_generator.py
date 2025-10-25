"""
response_generator.py

Module for generating responses using OpenAI-compatible API.
"""
from typing import Any, Dict, List

import requests
from requests.exceptions import RequestException, HTTPError


class ResponseAPIError(Exception):
    """Custom exception for response API errors."""
    def __init__(self, message: str, status_code: int | None = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class ResponseGenerator:
    def __init__(self, api_base: str | None = None, api_key: str | None = None):
        """Initialize the response generator with API configuration."""
        self.api_base = api_base
        self.api_key = api_key

    def get_responses(self, texts: List[str], model: str = "deepseek/deepseek-r1-0528-qwen3-8b") -> List[Dict[str, Any]]:
        """Call the responses endpoint to get model responses for each text.
        
        Args:
            texts: List of text strings to get responses for
            model: Name of the model to use for responses
            
        Returns:
            List of response data dictionaries containing:
            - text: The actual response text
            - id: Response ID
            - created_at: Timestamp
            - model: Model used
            - usage: Token usage stats
            
        Raises:
            ResponseAPIError: If the API returns an error, invalid format, or unexpected status
            RequestException: For network/connection errors
            ValueError: For invalid input
        """
        if not texts:
            raise ValueError("No texts provided for responses")
            
        # Construct the endpoint URL
        base = self.api_base.rstrip("/") if self.api_base else "http://localhost:5000/v1"
        url = f"{base}/responses"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else ""
        }
        
        data = {
            "input": texts,
            "model": model
        }
        
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=900)
            
            # Check for HTTP errors
            try:
                resp.raise_for_status()
            except HTTPError as e:
                # Try to get error details from response
                error_body = None
                try:
                    error_body = resp.json()
                except Exception:
                    pass
                raise ResponseAPIError(
                    f"Response API error: {str(e)}", 
                    status_code=resp.status_code,
                    response_body=error_body
                ) from e
                
            # Parse response
            try:
                result = resp.json()
            except ValueError as e:
                raise ResponseAPIError(
                    "Invalid JSON response from API", 
                    status_code=resp.status_code,
                    response_body=resp.text
                ) from e
                
            # Validate response format
            if not isinstance(result, dict):
                raise ResponseAPIError(
                    "Expected JSON object response",
                    status_code=resp.status_code,
                    response_body=result
                )
                
            output = result.get("output", [])
            if not isinstance(output, list):
                raise ResponseAPIError(
                    "Expected 'output' to be a list",
                    status_code=resp.status_code,
                    response_body=result
                )
            
            # Extract message content from output
            response_text = None
            for item in output:
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content", [])
                    for c in content:
                        if c.get("type") == "output_text":
                            response_text = c.get("text", "").strip()
                            break
                    if response_text:
                        break
            
            if not response_text:
                raise ResponseAPIError(
                    "No valid message response found in output",
                    status_code=resp.status_code,
                    response_body=result
                )
            
            # Return response with metadata
            response_data = {
                "text": response_text,
                "id": result.get("id"),
                "created_at": result.get("created_at"),
                "model": result.get("model"),
                "usage": result.get("usage", {}),
            }
            
            return [response_data]
        except RequestException as e:
            # Handle network/connection errors
            raise ResponseAPIError(
                f"Request failed: {str(e)}"
            ) from e