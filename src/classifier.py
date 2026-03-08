import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from src.utils import parse_llm_response
from src.prompts import PromptManager

# Load environment variables
load_dotenv()

class IntentClassifier:
    """Classifies user intent using an LLM"""
    
    # Available intents from our prompts
    VALID_INTENTS = ["code", "data", "writing", "career", "unclear"]
    
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        """
        Initialize the classifier with OpenAI client
        
        Args:
            model: The LLM model to use (default: gpt-3.5-turbo for speed/cost)
            api_key: OpenAI API key (if None, tries to get from env)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file")
        
        self.client = OpenAI(api_key=self.api_key)
        self.prompt_manager = PromptManager()
        
        # Create the classifier prompt template
        self.classifier_prompt = self._create_classifier_prompt()
    
    def _create_classifier_prompt(self) -> str:
        """
        Create the prompt for intent classification.
        This prompt is engineered to be:
        - Clear and specific about the task
        - Constrained to return only JSON
        - Explicit about the possible intents
        """
        valid_intents_str = ", ".join(self.VALID_INTENTS)
        
        prompt = f"""You are an intent classification system. Your task is to analyze user messages and classify them into one of these categories: {valid_intents_str}.

Classification Guidelines:
- **code**: User is asking about programming, debugging, writing code, or technical implementation
- **data**: User is asking about data analysis, statistics, datasets, numbers, or data visualization
- **writing**: User is asking for help improving text, writing style, grammar, or clarity
- **career**: User is asking about career advice, job searching, interviews, or professional development
- **unclear**: User's message is ambiguous, off-topic, or doesn't clearly fit any category

Important Rules:
1. Return ONLY a JSON object, no other text
2. The JSON must have exactly two fields: "intent" and "confidence"
3. "confidence" must be a float between 0.0 and 1.0
4. If uncertain, lean toward "unclear" rather than guessing

Examples:
User: "how do i sort a list in python?"
Response: {{"intent": "code", "confidence": 0.95}}

User: "what's the average of 10,20,30?"
Response: {{"intent": "data", "confidence": 0.98}}

User: "make this sentence better: it was good"
Response: {{"intent": "writing", "confidence": 0.85}}

User: "should I quit my job?"
Response: {{"intent": "career", "confidence": 0.90}}

User: "tell me a joke"
Response: {{"intent": "unclear", "confidence": 0.75}}

Now, classify this user message:
User: {{user_message}}

Response:"""
        
        return prompt
    
    def classify_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Classify the intent of a user message
        
        Args:
            user_message: The message from the user
            
        Returns:
            Dictionary with 'intent' and 'confidence' keys
            Defaults to {'intent': 'unclear', 'confidence': 0.0} on error
        """
        try:
            # Format the prompt with the user's message
            formatted_prompt = self.classifier_prompt.format(
                user_message=user_message
            )
            
            # Make the LLM call - using gpt-3.5-turbo for speed and cost
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise intent classifier. You always respond with valid JSON only."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,  # Low temperature for consistent classification
                max_tokens=100     # Keep it short and fast
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response (handles malformed JSON gracefully)
            result = parse_llm_response(response_text)
            
            # Validate that the intent is one of our valid intents
            if result["intent"] not in self.VALID_INTENTS:
                print(f"Warning: Invalid intent '{result['intent']}' received, defaulting to 'unclear'")
                result = {"intent": "unclear", "confidence": 0.0}
            
            # Ensure confidence is within bounds
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            
            return result
            
        except Exception as e:
            # Log the error and return safe default
            print(f"Error in classify_intent: {str(e)}")
            return {"intent": "unclear", "confidence": 0.0}
    
    def classify_with_fallback(self, user_message: str, threshold: float = 0.6) -> Dict[str, Any]:
        """
        Classify intent with confidence threshold fallback
        
        Args:
            user_message: The message from the user
            threshold: Minimum confidence to accept the classification
            
        Returns:
            If confidence >= threshold, returns the classification
            If confidence < threshold, returns {'intent': 'unclear', 'confidence': 0.0}
        """
        result = self.classify_intent(user_message)
        
        if result["confidence"] < threshold:
            return {"intent": "unclear", "confidence": 0.0}
        
        return result


# Standalone function version (for requirement compliance)
def classify_intent(message: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to classify intent
    This matches the requirement in the project spec
    
    Args:
        message: User message to classify
        api_key: Optional API key (if not provided, uses env)
        
    Returns:
        Dictionary with intent and confidence
    """
    classifier = IntentClassifier(api_key=api_key)
    return classifier.classify_intent(message)