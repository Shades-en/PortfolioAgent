from ai_server.schemas import User


class UserService:
    @classmethod
    async def get_user(cls, user_id: str | None = None, cookie_id: str | None = None) -> User | None:
        """
        Get a user by their ID or cookie ID.
        
        Args:
            user_id: MongoDB document ID of the user (optional)
            cookie_id: Cookie ID of the user (optional)
            
        Returns:
            User document if found, None otherwise
            
        Raises:
            ValueError: If neither user_id nor cookie_id is provided
        """
        if not user_id and not cookie_id:
            raise ValueError("Either user_id or cookie_id must be provided")
        
        return await User.get_by_id_or_cookie(user_id=user_id, cookie_id=cookie_id)
