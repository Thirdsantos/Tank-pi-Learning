"""Configuration module for the aqua learning project."""

from .database import get_supabase_client, is_supabase_available

__all__ = ['get_supabase_client', 'is_supabase_available']
