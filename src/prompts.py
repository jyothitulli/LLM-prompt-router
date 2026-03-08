import json
import os
from pathlib import Path

class PromptManager:
    """Manages loading and accessing system prompts"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            # Default to config/prompts.json relative to project root
            current_dir = Path(__file__).parent.parent
            config_path = current_dir / "config" / "prompts.json"
        
        self.config_path = config_path
        self.prompts = self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from JSON configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                return data
        except FileNotFoundError:
            print(f"Warning: Prompts file not found at {self.config_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in prompts file at {self.config_path}")
            return {}
    
    def get_prompt(self, intent):
        """Get system prompt for a specific intent"""
        prompt_data = self.prompts.get(intent, {})
        return prompt_data.get("system_prompt", "")
    
    def get_all_intents(self):
        """Get list of all available intents"""
        return list(self.prompts.keys())
    
    def is_valid_intent(self, intent):
        """Check if an intent is valid"""
        return intent in self.prompts
    
    def get_prompt_info(self, intent):
        """Get full prompt info including name and system prompt"""
        return self.prompts.get(intent, {})