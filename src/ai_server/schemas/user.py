from __future__ import annotations

import pymongo
from beanie import Document
from bson import ObjectId

from datetime import datetime, timezone
from pydantic import Field
from enum import Enum

from ai_server.api.exceptions.db_exceptions import UserRetrievalFailedException


class UserType(Enum):
    GUEST = "guest"
    USER = "logged_in"


class User(Document):
    cookie_id: str
    category: UserType = UserType.GUEST
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "users"
        indexes = [
            [("cookie_id", pymongo.ASCENDING)]
        ]
    
    @classmethod
    async def get_by_id(cls, user_id: str) -> User | None:
        """Retrieve a user by their MongoDB document ID."""
        try:
            return await cls.get(ObjectId(user_id))
        except Exception as e:
            raise UserRetrievalFailedException(
                message="Failed to retrieve user by ID",
                note=f"user_id={user_id}, error={str(e)}"
            )
    
    @classmethod
    async def get_by_cookie_id(cls, cookie_id: str) -> User | None:
        """Retrieve a user by their cookie ID."""
        try:
            return await cls.find_one(cls.cookie_id == cookie_id)
        except Exception as e:
            raise UserRetrievalFailedException(
                message="Failed to retrieve user by cookie ID",
                note=f"cookie_id={cookie_id}, error={str(e)}"
            )
    
    @classmethod
    async def get_by_id_or_cookie(cls, user_id: str | None, cookie_id: str) -> User | None:
        """
        Retrieve a user by ID or cookie ID.
        
        Args:
            user_id: MongoDB document ID of the user (optional)
            cookie_id: Cookie ID of the user
            
        Returns:
            User object if found, None otherwise
        """
        if user_id:
            return await cls.get_by_id(user_id)
        else:
            return await cls.get_by_cookie_id(cookie_id)