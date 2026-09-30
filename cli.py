"""
Terminal CLI Interface for Aether Chatbot.
Run with: python cli.py
"""

import sys
import uuid
from app.config import settings
from app.context_manager import context_manager
from app.llm_client import llm_client, LLMClientError
from app.persona import DEFAULT_BOT_NAME

# Terminal ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

def print_banner():
    print(f"{CYAN}{BOLD}================================================================{RESET}")
    print(f"{CYAN}{BOLD}   ✨ {DEFAULT_BOT_NAME} - Context-Aware Terminal Chatbot{RESET}")
    print(f"{CYAN}{BOLD}   Model: {settings.GEMINI_MODEL} | Strategy: {settings.CONTEXT_STRATEGY}{RESET}")
    print(f"{CYAN}{BOLD}================================================================{RESET}")
    print(f"{DIM}Commands: /reset (clear memory), /stats (context info), /exit (quit){RESET}\n")

def run_cli():
    print_banner()

    # Pre-check API key
    if not settings.is_api_key_configured():
        print(f"{RED}{BOLD}[!] Configuration Warning:{RESET}")
        print(f"{RED}No valid GEMINI_API_KEY detected in .env file.{RESET}")
        print(f"{YELLOW}Please create or edit '.env' with your key:{RESET}")
        print(f"  GEMINI_API_KEY=your_key_here\n")
        print(f"{DIM}(You can get a free key at: https://aistudio.google.com/app/apikey){RESET}\n")

    session_id = str(uuid.uuid4())
    conversation = context_manager.get_or_create(session_id)

    print(f"{DIM}Active Session ID: {session_id[:8]}... (Context window: {conversation.max_messages} msgs){RESET}\n")

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{CYAN}Goodbye! Session ended.{RESET}")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print(f"{CYAN}Goodbye! Session ended.{RESET}")
            break

        if user_input.lower() == "/reset":
            conversation.clear()
            print(f"{YELLOW}Conversation memory cleared. Starting fresh!{RESET}\n")
            continue

        if user_input.lower() == "/stats":
            stats = conversation.get_stats()
            print(f"\n{CYAN}--- Context Window Telemetry ---{RESET}")
            print(f"  Strategy: {stats['strategy']}")
            print(f"  Total messages stored: {stats['total_messages']}")
            print(f"  Messages in active window: {stats['messages_in_context']}")
            print(f"  Messages pruned/summarized: {stats['messages_pruned_or_summarized']}")
            if stats.get('running_summary_preview'):
                print(f"  Running Summary: {stats['running_summary_preview']}")
            print(f"{CYAN}--------------------------------{RESET}\n")
            continue

        if user_input.lower().startswith("/strategy"):
            parts = user_input.split()
            if len(parts) > 1 and parts[1].lower() in ("sliding_window", "summarization"):
                conversation.strategy = parts[1].lower()
                print(f"{YELLOW}Strategy switched to: {conversation.strategy}{RESET}\n")
            else:
                print(f"{YELLOW}Usage: /strategy sliding_window  OR  /strategy summarization{RESET}\n")
            continue

        # Generate response
        print(f"{DIM}Thinking...{RESET}", end="\r", flush=True)

        try:
            reply, stats = llm_client.generate_response(
                conversation=conversation,
                user_message=user_input
            )
            # Clear "Thinking..." line
            print(" " * 20, end="\r")
            
            # Print response
            print(f"{CYAN}{BOLD}{DEFAULT_BOT_NAME} >{RESET}\n{reply}\n")
            
            # Sub-telemetry note
            in_win = stats['messages_in_context']
            total = stats['total_messages']
            print(f"{DIM}[Context: {in_win}/{total} msgs in window | Strategy: {stats['strategy']}]{RESET}\n")

        except LLMClientError as e:
            print(" " * 20, end="\r")
            print(f"{RED}{BOLD}[Error] {e.message}{RESET}")
            if e.details and e.details != e.message:
                print(f"{DIM}{e.details}{RESET}\n")
            else:
                print()
        except Exception as e:
            print(" " * 20, end="\r")
            print(f"{RED}{BOLD}[Unexpected Error]: {e}{RESET}\n")

if __name__ == "__main__":
    run_cli()
