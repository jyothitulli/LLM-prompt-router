#!/usr/bin/env python
"""
Quick demo of the prompt router
Run with: python demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.classifier import IntentClassifier
from src.router import PromptRouter

def demo():
    """Demonstrate the prompt router with examples"""
    
    print("\n" + "🌟" * 50)
    print("🌟  LLM-POWERED PROMPT ROUTER - QUICK DEMO  🌟")
    print("🌟" * 50 + "\n")
    
    classifier = IntentClassifier()
    router = PromptRouter()
    
    # Demo examples
    examples = [
        ("💻 Coding", "how do i sort a list of dictionaries by a key in python?"),
        ("📊 Data", "what's the median of these numbers: 12, 45, 67, 23, 89, 34, 56?"),
        ("✍️ Writing", "make this sentence better: 'The meeting was really good and we talked about many things.'"),
        ("💼 Career", "what should I include in my resume for a software engineer position?"),
        ("❓ Unclear", "tell me a story about a dragon")
    ]
    
    for category, message in examples:
        print(f"\n{category}")
        print(f"📝 User: {message}")
        
        # Classify
        intent_data = classifier.classify_intent(message)
        print(f"🤔 Intent: {intent_data['intent']} (confidence: {intent_data['confidence']:.2f})")
        
        # Respond
        response = router.route_and_respond(message, intent_data)
        print(f"💬 Assistant: {response[:150]}..." if len(response) > 150 else f"💬 Assistant: {response}")
        print("-" * 70)

if __name__ == "__main__":
    demo()