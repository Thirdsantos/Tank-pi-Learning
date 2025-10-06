"""Database configuration and connection setup."""

from dotenv import load_dotenv
import os
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY environment variables")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def is_supabase_available() -> bool:
    """Check if Supabase credentials are available."""
    return bool(SUPABASE_URL and SUPABASE_KEY)
