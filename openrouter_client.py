"""
OpenRouter Client - Unified LLM interface supporting multiple models
Supports Claude, GPT-4o, Gemini, and other models via OpenRouter API
"""

import os
import time
import json
import base64
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import requests
from dotenv import load_dotenv

# Load environment variables from .env file (e.g., API keys)
load_dotenv()


class OpenRouterClient:
    """
    Unified client for accessing multiple LLM providers through OpenRouter.

    Supports:
    - Text generation
    - Vision/multimodal (PDF, images)
    - Structured JSON output
    - Multiple model providers (Claude, GPT-4o, Gemini, etc.)
    """

    # Shorthand mappings to full model identifiers
    # Allows users to write "claude-sonnet" instead of "anthropic/claude-sonnet-4.5"
    MODELS = {
        "claude-sonnet": "anthropic/claude-sonnet-4.5",
        "claude-opus": "anthropic/claude-opus-4.5",
        "gpt-5.2": "openai/gpt-5.2-pro",
        "gemini-pro": "google/gemini-3-pro-preview",  # Only a preview, should we still benchmark on it? TODO
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-opus-4.5",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """
        Initialize OpenRouter client.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model identifier (use MODELS dict or full model path)
            base_url: OpenRouter API endpoint
        """
        # Use provided key, or fall back to environment variable
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY environment variable."
            )

        # dict.get() returns the value if key exists, otherwise returns the key itself
        # This allows both "claude-sonnet" and "anthropic/claude-sonnet-4.5" to work
        self.model = self.MODELS.get(model, model)
        self.base_url = base_url

        print(f"OpenRouterClient initialized with model: {self.model}")

    def _parse_json_from_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown code blocks."""
        # Try direct parse first (response might be pure JSON)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # LLMs often wrap JSON in markdown code blocks like ```json ... ```
        # These regex patterns extract the content between the backticks
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # Matches ```json ... ```
            r"```geojson\s*([\s\S]*?)\s*```",  # Matches ```geojson ... ```
        ]

        for pattern in patterns:
            # re.search finds the first match anywhere in the string
            match = re.search(pattern, content)
            if match:
                try:
                    # group(1) returns the first captured group (content inside parentheses)
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue

        # If no JSON is found, return None
        return None

    def chat_with_pdf(
        self,
        pdf_path: str,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Send a prompt along with a PDF document.

        Args:
            pdf_path: Path to PDF file
            prompt: Question or instruction about the PDF
            system_message: Optional system instruction
            max_tokens: Maximum response length

        Returns:
            Response dict with 'content', 'model', 'tokens', 'processing_time', 'success'
        """
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Read PDF as binary and encode to base64 string
        # Base64 encoding converts binary data to ASCII text, required for JSON transport
        with open(pdf_path, "rb") as pdf_file:
            pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")

        pdf_file_name = os.path.basename(pdf_path)

        # Data URI format: tells the API this is a base64-encoded PDF
        # Format: data:[media-type];base64,[data]
        pdf_data = f"data:application/pdf;base64,{pdf_data}"

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})

        # Multimodal message format: array of content blocks with different types
        # This allows mixing text and files in a single message
        user_content = [
            {"type": "text", "text": prompt},
            {
                "type": "file",
                "file": {"filename": pdf_file_name, "file_data": pdf_data},
            },
        ]
        messages.append({"role": "user", "content": user_content})

        return self._send_completion(
            messages=messages,
            max_tokens=max_tokens,
        )

    def _send_completion(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Send a chat completion request to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        start_time = time.time()

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,  # Automatically serializes dict to JSON
                timeout=120,  # Seconds before request times out
            )
            # Raises HTTPError for 4xx/5xx status codes
            response.raise_for_status()
            data = response.json()

            # OpenRouter follows OpenAI's response format:
            # choices[0].message.content contains the model's text response
            content = data["choices"][0]["message"]["content"]

            result = {
                "content": content,
                "model": data.get("model", self.model),
                "tokens": data.get("usage", {}),
                "processing_time": time.time() - start_time,
                "success": True,
            }

            # Attempt to extract structured JSON from the response
            parsed = self._parse_json_from_response(content)
            if parsed:
                result["parsed_json"] = parsed
            else:
                result["json_error"] = "Could not extract valid JSON from response"
                result["success"] = False

            return result

        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    def extract_geojson_from_pdf(
        self, pdf_path: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract planning area boundary as GeoJSON from PDF.

        Args:
            pdf_path: Path to planning document PDF
            context: Additional context about the document

        Returns:
            Dict with GeoJSON and metadata
        """
        # f-string with conditional: only includes context line if context is provided
        prompt = f"""You are a GIS analyst specializing in UK planning documents.

                    Analyze this planning document and extract the geographic boundary of the planning area as GeoJSON.

                    Task:
                    1. Identify if the document contains a map showing the planning area boundary
                    2. Look for coordinate information (latitude/longitude, OS grid references, or textual descriptions)
                    3. Extract or infer the boundary polygon(s)

                    Requirements:
                    - Output a valid GeoJSON Feature with MultiPolygon geometry
                    - Always use "MultiPolygon" type (even for single connected areas, wrap it in MultiPolygon format)
                    - Coordinates must be in [longitude, latitude] format (WGS84)
                    - If the document shows multiple disconnected areas, include all of them in the MultiPolygon
                    - If the document lacks precise coordinates, make a reasonable estimate based on:
                      * Street names and landmarks mentioned
                      * The map boundaries if visible
                      * The administrative area description

                    {f"Additional context: {context}" if context else ""}

                    Output only valid GeoJSON in this exact format:
                    {{
                    "type": "Feature",
                    "geometry": {{
                        "type": "MultiPolygon",
                        "coordinates": [
                            [[[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon1, lat1]]],
                            [[[lon4, lat4], [lon5, lat5], [lon6, lat6], [lon4, lat4]]]
                        ]
                    }},
                    "properties": {{
                        "source": "planning_document",
                        "confidence": "high|medium|low",
                        "method": "description of extraction method"
                    }}
                    }}

                    Notes:
                    - For a single connected area, use one polygon in the MultiPolygon array
                    - For multiple disconnected areas, use multiple polygons in the array
                    - Each polygon must close (first and last coordinate pairs must be identical)"""

        system_message = (
            "You are a GIS analyst. Always respond with valid GeoJSON only."
        )

        result = self.chat_with_pdf(
            pdf_path=pdf_path,
            prompt=prompt,
            system_message=system_message,
            max_tokens=8192,
        )

        return result


# Example usage
if __name__ == "__main__":
    client = OpenRouterClient(model="claude-sonnet")

    response = client.extract_geojson_from_pdf("path_to_planning_document.pdf")
    print("Response:", response)

    if response.get("success") and response.get("parsed_json"):
        geojson = response["parsed_json"]

        with open("output.geojson", "w") as f:
            json.dump(geojson, f, indent=2)
        print("GeoJSON saved to output.geojson")
    else:
        print("Failed to extract GeoJSON:")
        # Use 'or' to try json_error first, fall back to error if not present
        print(response.get("json_error") or response.get("error"))
        print("\nRaw response:")
        # Provide empty string as default if 'content' key doesn't exist
        print(response.get("content", ""))
