#!/usr/bin/env python3
"""
Conversation History Cache using Redis
"""

import json
import redis
from typing import List, Dict, Optional
from datetime import timedelta


class ConversationCache:
    """Redis-based conversation history cache"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        """
        Initialize Redis client
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
        """
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = 3600  # 1 hour default TTL
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a message to conversation history
        
        Args:
            session_id: Session identifier
            role: Message role (user/assistant/system)
            content: Message content
            metadata: Optional metadata
        """
        message = {
            'role': role,
            'content': content,
            'metadata': metadata or {}
        }
        
        key = f"conversation:{session_id}"
        self.client.rpush(key, json.dumps(message))
        self.client.expire(key, self.ttl)
    
    def get_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """
        Get conversation history for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of messages
        """
        key = f"conversation:{session_id}"
        messages = self.client.lrange(key, -limit, -1)
        
        return [json.loads(msg) for msg in messages]
    
    def clear_history(self, session_id: str) -> None:
        """Clear conversation history for a session"""
        key = f"conversation:{session_id}"
        self.client.delete(key)
    
    def set_session_ttl(self, session_id: str, ttl: int) -> None:
        """
        Set TTL for a session
        
        Args:
            session_id: Session identifier
            ttl: Time to live in seconds
        """
        key = f"conversation:{session_id}"
        self.client.expire(key, ttl)
    
    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs"""
        pattern = "conversation:*"
        keys = self.client.keys(pattern)
        return [key.replace("conversation:", "") for key in keys]
    
    def delete_session(self, session_id: str) -> None:
        """Delete a session entirely"""
        key = f"conversation:{session_id}"
        self.client.delete(key)


if __name__ == "__main__":
    # Test conversation cache
    cache = ConversationCache()
    
    session_id = "test_session"
    
    # Add messages
    cache.add_message(session_id, "user", "Hello, how are you?")
    cache.add_message(session_id, "assistant", "I'm doing well, thank you!")
    cache.add_message(session_id, "user", "What can you help me with?")
    
    # Retrieve history
    history = cache.get_history(session_id)
    
    print("Conversation History:")
    for msg in history:
        print(f"{msg['role']}: {msg['content']}")
