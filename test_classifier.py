#!/usr/bin/env python
"""
Quick test script to verify the classifier works
Run with: python test_classifier.py
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import classify_intent
from tests.test_messages import get_test_messages

def test_classifier():
    """Test the classifier with a few sample messages"""
    
    print("=" * 60)
    print("TESTING INTENT CLASSIFIER")
    print("=" * 60)
    
    # Get a few test messages
    test_messages = get_test_messages()[:10]  # First 10 messages
    
    for msg in test_messages:
        print(f"\n📝 Message: {msg}")
        
        try:
            result = classify_intent(msg)
            intent = result.get("intent", "unknown")
            confidence = result.get("confidence", 0.0)
            
            # Visual indicator based on intent
            if intent == "code":
                emoji = "💻"
            elif intent == "data":
                emoji = "📊"
            elif intent == "writing":
                emoji = "✍️"
            elif intent == "career":
                emoji = "💼"
            else:
                emoji = "❓"
            
            print(f"{emoji} Intent: {intent} (confidence: {confidence:.2f})")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Check for API key
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        print("Please add your API key to .env first:")
        print('echo "OPENAI_API_KEY=your-key-here" > .env')
        sys.exit(1)
    
    test_classifier()