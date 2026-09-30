"""
Automated Multi-Turn Conversation & Context Window Test Suite.
Verifies:
1. Unit tests for context window truncation (Sliding Window)
2. Unit tests for Progressive Summarization triggers
3. Session isolation between different conversation IDs
4. Live 5-turn conversational memory test against Gemini API (if key is set)

Usage:
  python test_conversation.py
"""

import sys
from app.config import settings
from app.context_manager import ContextManager, Conversation
from app.persona import get_system_prompt

def test_sliding_window_logic():
    print("\n[Test 1] Testing Sliding Window Truncation Logic...")
    cm = ContextManager()
    conv = cm.get_or_create("test-sliding", strategy="sliding_window", max_messages=4)

    # Add 6 messages (3 user, 3 model)
    conv.add_message("user", "Message 1 (User)")
    conv.add_message("model", "Reply 1 (Bot)")
    conv.add_message("user", "Message 2 (User)")
    conv.add_message("model", "Reply 2 (Bot)")
    conv.add_message("user", "Message 3 (User)")
    conv.add_message("model", "Reply 3 (Bot)")

    stats = conv.get_stats()
    assert stats["total_messages"] == 6, f"Expected 6 total, got {stats['total_messages']}"
    assert stats["messages_in_context"] == 4, f"Expected 4 in context, got {stats['messages_in_context']}"
    assert stats["messages_pruned_or_summarized"] == 2, f"Expected 2 pruned, got {stats['messages_pruned_or_summarized']}"

    window_context = conv.prepare_sliding_window_context()
    # Ensure window has at most 4 items and starts with user
    assert len(window_context) <= 4, f"Window size exceeded: {len(window_context)}"
    assert window_context[0]["role"] == "user", "Window must start with 'user' role"
    print("  --> PASS: Sliding window properly capped context at 4 and preserved 'user' lead.")


def test_session_isolation():
    print("\n[Test 2] Testing Multi-Session Isolation...")
    cm = ContextManager()
    conv_a = cm.get_or_create("session-alice")
    conv_b = cm.get_or_create("session-bob")

    conv_a.add_message("user", "My secret code is ALPHA")
    conv_b.add_message("user", "My secret code is BRAVO")

    assert len(conv_a.messages) == 1
    assert len(conv_b.messages) == 1
    assert "ALPHA" in conv_a.messages[0].content
    assert "BRAVO" in conv_b.messages[0].content
    assert "ALPHA" not in conv_b.messages[0].content
    print("  --> PASS: Separate conversation IDs maintain completely isolated histories.")


def test_summarization_thresholds():
    print("\n[Test 3] Testing Summarization Trigger Conditions...")
    cm = ContextManager()
    conv = cm.get_or_create("test-sum", strategy="summarization", max_messages=4)

    # 4 messages: should not trigger summarization yet
    for i in range(4):
        conv.add_message("user" if i % 2 == 0 else "model", f"Content {i}")
    assert not conv.needs_summarization(), "Should not trigger when <= max_messages"

    # 5th message: should trigger
    conv.add_message("user", "Trigger message")
    assert conv.needs_summarization(), "Should trigger when > max_messages"

    to_sum = conv.get_messages_to_summarize()
    assert len(to_sum) > 0, "Should have messages earmarked for summarization"

    # Commit summary
    conv.commit_summary("User discussed initial topics 0-1", len(to_sum))
    assert conv.running_summary == "User discussed initial topics 0-1"
    print("  --> PASS: Summarization triggering and state updates function properly.")



def run_live_multi_turn_test():
    print("\n[Test 4] Live 5-Turn Gemini API Multi-Turn Test...")
    from app.llm_client import llm_client

    if not settings.is_api_key_configured():
        print("  --> SKIPPED: GEMINI_API_KEY is not configured in .env yet.")
        print("  --> To run live test, add your key to .env and run this script again.")
        return

    print("  API Key detected. Running live 5-turn dialogue...")
    cm = ContextManager()
    conv = cm.get_or_create("live-test-session", strategy="sliding_window", max_messages=10)

    turns = [
        ("Turn 1 (Introduce Identity & Project)", "Hi Aether! My name is Jordan and I am developing an autonomous drone in Python."),
        ("Turn 2 (Recall Test)", "What is my name and what project am I working on?"),
        ("Turn 3 (Domain Context)", "What are two critical sensors I should attach to this drone for obstacle avoidance?"),
        ("Turn 4 (Cross-Turn Correlation)", "Why is the programming language I mentioned earlier suitable for integrating those sensors?"),
        ("Turn 5 (Multi-Turn Synthesis)", "Please give a 2-bullet summary of our discussion so far.")
    ]

    import time
    for turn_label, prompt in turns:
        print(f"\n  >> {turn_label}: '{prompt}'")
        reply, stats = llm_client.generate_response(conv, prompt)
        print(f"  << Aether: {reply[:150]}... [length: {len(reply)} chars]")
        print(f"     [Stats: {stats['messages_in_context']}/{stats['total_messages']} messages in context]")
        time.sleep(1.5)

    # Check that Jordan was recalled
    all_bot_replies = " ".join([m.content for m in conv.messages if m.role == "model"])
    assert "jordan" in all_bot_replies.lower(), "Model should have remembered user's name Jordan"
    assert "drone" in all_bot_replies.lower(), "Model should have remembered the drone project"
    assert "python" in all_bot_replies.lower(), "Model should have remembered Python"

    print("\n  --> PASS: Live 5-turn conversation successfully maintained context throughout all turns!")


if __name__ == "__main__":
    print("==================================================")
    print(" Running Context-Aware Chatbot Verification Tests")
    print("==================================================")

    test_sliding_window_logic()
    test_session_isolation()
    test_summarization_thresholds()
    run_live_multi_turn_test()

    print("\n==================================================")
    print(" ALL LOCAL TESTS COMPLETED SUCCESSFULLY! ")
    print("==================================================")
