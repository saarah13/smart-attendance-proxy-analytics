"""
modules/db.py
Shared MongoDB connection - single client reused across all modules.
Import get_db() instead of creating MongoClient in every module.
"""
from pymongo import MongoClient
import config

_client = None


def get_db():
    """Return the shared MongoDB database handle (lazy-initialised)."""
    global _client
    if _client is None:
        _client = MongoClient(config.MONGO_URI)
    return _client[config.DB_NAME]


def close_db():
    """Explicitly close the shared client (call on app shutdown if needed)."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
