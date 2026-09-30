import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file in project root
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    """Application configuration loaded from environment variables."""
    
    # Gemini API Credentials
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
    
    # Context Window Strategy: 'sliding_window' or 'summarization'
    CONTEXT_STRATEGY: str = os.getenv("CONTEXT_STRATEGY", "sliding_window").strip().lower()
    
    # Maximum messages to keep in direct context window
    MAX_CONTEXT_MESSAGES: int = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))
    
    # Server settings
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    @classmethod
    def reload(cls):
        """Reload configuration from disk."""
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        cls.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
        cls.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
        cls.CONTEXT_STRATEGY = os.getenv("CONTEXT_STRATEGY", "sliding_window").strip().lower()
        cls.MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "10"))
        cls.HOST = os.getenv("HOST", "127.0.0.1")
        cls.PORT = int(os.getenv("PORT", "8000"))

    @classmethod
    def is_api_key_configured(cls) -> bool:
        """Check if a real API key is configured (not empty or default placeholder)."""
        key = cls.GEMINI_API_KEY
        if not key or key == "your_gemini_api_key_here" or "your_" in key.lower():
            return False
        return True

settings = Settings()
