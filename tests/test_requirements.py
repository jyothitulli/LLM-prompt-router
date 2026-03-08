#!/usr/bin/env python
"""
Comprehensive test suite for verifying all project requirements
Run with: python -m tests.test_requirements
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import IntentClassifier, classify_intent
from src.router import PromptRouter, route_and_respond
from src.prompts import PromptManager
from src.utils import Logger, parse_llm_response
from tests.test_messages import get_test_messages, get_categorized_messages

class RequirementTester:
    """Tests all core requirements"""
    
    def __init__(self):
        self.classifier = IntentClassifier()
        self.router = PromptRouter()
        self.prompt_manager = PromptManager()
        self.logger = Logger()
        self.tests_passed = 0
        self.tests_failed = 0
        
    def print_header(self, title):
        """Print a formatted header"""
        print("\n" + "=" * 70)
        print(f"📋 {title}")
        print("=" * 70)
    
    def print_result(self, test_name, passed, details=""):
        """Print test result with emoji"""
        if passed:
            print(f"✅ PASS: {test_name}")
            self.tests_passed += 1
        else:
            print(f"❌ FAIL: {test_name}")
            self.tests_failed += 1
        if details:
            print(f"   {details}")
    
    def test_requirement_1_system_prompts(self):
        """Requirement 1: At least 4 distinct expert system prompts"""
        self.print_header("Requirement 1: System Prompts")
        
        # Get all prompts
        prompts = self.prompt_manager.prompts
        prompt_count = len(prompts)
        
        # Check count
        passed_count = prompt_count >= 4
        self.print_result(
            "At least 4 prompts exist",
            passed_count,
            f"Found {prompt_count} prompts"
        )
        
        # Check each prompt has required structure
        all_prompts_valid = True
        for intent, prompt_data in prompts.items():
            has_name = "name" in prompt_data
            has_system_prompt = "system_prompt" in prompt_data
            prompt_valid = has_name and has_system_prompt
            
            if not prompt_valid:
                all_prompts_valid = False
                print(f"   ⚠️  Invalid prompt for intent '{intent}': missing name or system_prompt")
            else:
                prompt_length = len(prompt_data["system_prompt"])
                print(f"   📝 {prompt_data['name']}: {prompt_length} characters")
        
        self.print_result(
            "All prompts have required structure",
            all_prompts_valid
        )
        
        # Check prompts are loaded from config, not hardcoded
        from src.prompts import PromptManager
        prompt_manager = PromptManager()
        config_loaded = prompt_manager.config_path.exists()
        
        self.print_result(
            "Prompts loaded from configuration file",
            config_loaded,
            f"Config path: {prompt_manager.config_path}"
        )
        
        return passed_count and all_prompts_valid and config_loaded
    
    def test_requirement_2_classify_intent_function(self):
        """Requirement 2: classify_intent returns structured JSON"""
        self.print_header("Requirement 2: classify_intent Function")
        
        test_messages = [
            "how do i sort a list in python?",
            "what's the average of 1,2,3?",
            "help me improve this sentence"
        ]
        
        all_valid = True
        for msg in test_messages:
            result = classify_intent(msg)
            
            # Check structure
            has_intent = "intent" in result
            has_confidence = "confidence" in result
            
            # Check types
            intent_valid = isinstance(result.get("intent"), str)
            confidence_valid = isinstance(result.get("confidence"), (int, float))
            
            is_valid = has_intent and has_confidence and intent_valid and confidence_valid
            
            if is_valid:
                print(f"   📝 '{msg[:30]}...' → {result['intent']} ({result['confidence']:.2f})")
            else:
                all_valid = False
                print(f"   ⚠️  Invalid result for '{msg}': {result}")
        
        self.print_result(
            "classify_intent returns valid structured JSON",
            all_valid
        )
        
        return all_valid
    
    def test_requirement_3_route_and_respond_function(self):
        """Requirement 3: route_and_respond uses intent to select prompt"""
        self.print_header("Requirement 3: route_and_respond Function")
        
        test_cases = [
            ("code", "write a function to add two numbers"),
            ("data", "what is the mean of 1,2,3,4,5?"),
            ("writing", "make this better: 'it was good'"),
            ("career", "how to prepare for interview")
        ]
        
        all_valid = True
        for intent, message in test_cases:
            intent_data = {"intent": intent, "confidence": 0.95}
            response = route_and_respond(message, intent_data)
            
            # Check response is a non-empty string
            is_valid = isinstance(response, str) and len(response) > 0
            
            if is_valid:
                preview = response[:50] + "..." if len(response) > 50 else response
                print(f"   📝 {intent}: {preview}")
            else:
                all_valid = False
                print(f"   ⚠️  Invalid response for {intent}")
        
        self.print_result(
            "route_and_respond generates valid responses",
            all_valid
        )
        
        return all_valid
    
    def test_requirement_4_unclear_intent_handling(self):
        """Requirement 4: Unclear intent generates clarification question"""
        self.print_header("Requirement 4: Unclear Intent Handling")
        
        unclear_messages = [
            "tell me a joke",
            "what's the meaning of life?",
            "hey how are you?"
        ]
        
        all_valid = True
        for msg in unclear_messages:
            intent_data = {"intent": "unclear", "confidence": 0.5}
            response = route_and_respond(msg, intent_data)
            
            # Check response is a question (contains ? or asks for clarification)
            is_question = "?" in response or "clarify" in response.lower() or "what" in response.lower()
            is_reasonable = len(response) > 10 and len(response) < 200  # Not too short or long
            
            if is_question and is_reasonable:
                print(f"   📝 '{msg}'\n      → {response[:80]}...")
            else:
                all_valid = False
                print(f"   ⚠️  Not a good clarification: '{response[:50]}...'")
        
        self.print_result(
            "Unclear intent generates clarification question",
            all_valid
        )
        
        return all_valid
    
    def test_requirement_5_logging(self):
        """Requirement 5: Interactions logged to route_log.jsonl"""
        self.print_header("Requirement 5: Logging")
        
        # Use a temporary log file for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            test_logger = Logger(log_dir=tmpdir)
            
            # Log a test interaction
            test_logger.log_interaction(
                user_message="test message",
                intent="test",
                confidence=0.95,
                final_response="test response"
            )
            
            # Read logs
            logs = test_logger.read_logs()
            
            # Check file exists
            file_exists = test_logger.log_file.exists()
            
            # Check log structure
            log_valid = False
            if logs:
                log = logs[0]
                required_keys = ["timestamp", "user_message", "intent", "confidence", "final_response"]
                has_all_keys = all(key in log for key in required_keys)
                log_valid = has_all_keys and log["user_message"] == "test message"
            
            self.print_result(
                "Log file created",
                file_exists,
                f"Log path: {test_logger.log_file}"
            )
            
            self.print_result(
                "Log entries have required structure",
                log_valid
            )
        
        # Check main logger
        main_logs = self.logger.read_logs(n_last=1)
        if main_logs:
            print(f"   📝 Latest log: {main_logs[0]['intent']} - {main_logs[0]['timestamp']}")
        
        return file_exists and log_valid
    
    def test_requirement_6_error_handling(self):
        """Requirement 6: Graceful handling of malformed JSON"""
        self.print_header("Requirement 6: Error Handling")
        
        # Test malformed JSON responses
        malformed_responses = [
            '{"intent": "code", "confidence": 0.95}',  # Valid
            '{"intent": "code" confidence: 0.95}',     # Malformed
            'The intent is code with confidence 0.95', # No JSON
            '{"intent": "code", "confidence": "high"}', # Wrong type
            '{"purpose": "coding"}'                     # Missing fields
        ]
        
        all_handled = True
        for i, response in enumerate(malformed_responses):
            result = parse_llm_response(response)
            
            # Should always return dict with intent and confidence
            has_intent = "intent" in result
            has_confidence = "confidence" in result
            
            if has_intent and has_confidence:
                print(f"   📝 Test {i+1}: → {result}")
            else:
                all_handled = False
                print(f"   ⚠️  Test {i+1}: Failed → {result}")
        
        self.print_result(
            "Malformed responses handled gracefully",
            all_handled
        )
        
        return all_handled
    
    def run_all_tests(self):
        """Run all requirement tests"""
        print("\n" + "🚀" * 35)
        print("🚀  RUNNING COMPREHENSIVE REQUIREMENT TESTS  🚀")
        print("🚀" * 35 + "\n")
        
        # Run all tests
        r1 = self.test_requirement_1_system_prompts()
        r2 = self.test_requirement_2_classify_intent_function()
        r3 = self.test_requirement_3_route_and_respond_function()
        r4 = self.test_requirement_4_unclear_intent_handling()
        r5 = self.test_requirement_5_logging()
        r6 = self.test_requirement_6_error_handling()
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 FINAL TEST SUMMARY")
        print("=" * 70)
        print(f"✅ Tests Passed: {self.tests_passed}")
        print(f"❌ Tests Failed: {self.tests_failed}")
        print(f"📈 Total Tests: {self.tests_passed + self.tests_failed}")
        
        if self.tests_failed == 0:
            print("\n🎉 ALL REQUIREMENTS VERIFIED! 🎉")
            print("Your prompt router is ready for submission!")
        else:
            print(f"\n⚠️  {self.tests_failed} requirements need attention.")
        
        print("=" * 70 + "\n")
        
        return self.tests_failed == 0

def test_with_sample_messages():
    """Test the router with all sample messages"""
    print("\n" + "📊" * 35)
    print("📊  TESTING WITH ALL SAMPLE MESSAGES  📊")
    print("📊" * 35 + "\n")
    
    classifier = IntentClassifier()
    router = PromptRouter()
    
    categorized = get_categorized_messages()
    
    results = {}
    for category, messages in categorized.items():
        print(f"\n📁 Category: {category.upper()}")
        print("-" * 50)
        
        category_results = []
        for msg in messages[:3]:  # Test first 3 from each category
            intent_data = classifier.classify_intent(msg)
            response = router.route_and_respond(msg, intent_data)
            
            correct = intent_data["intent"] == category or \
                     (category in ["ambiguous", "unclear"] and intent_data["intent"] in ["unclear", "ambiguous"])
            
            status = "✅" if correct else "❌"
            print(f"{status} '{msg[:40]}...' → {intent_data['intent']} ({intent_data['confidence']:.2f})")
            
            category_results.append(correct)
        
        results[category] = category_results
    
    return results

if __name__ == "__main__":
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)
    
    # Run requirement tests
    tester = RequirementTester()
    all_passed = tester.run_all_tests()
    
    # Optional: Test with all sample messages
    test_with_sample_messages()