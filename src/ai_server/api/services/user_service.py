from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.schemas import User
from ai_server.utils.tracing import trace_operation


class UserService:
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
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
        
        Traced as CHAIN span for service-level orchestration.
        """
        if not user_id and not cookie_id:
            raise ValueError("Either user_id or cookie_id must be provided")
        
        return await User.get_by_id_or_cookie(user_id=user_id, cookie_id=cookie_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_user(
        cls, 
        user_id: str | None = None, 
        cookie_id: str | None = None,
        cascade: bool = True
    ) -> dict:
        """
        Delete a user by their ID or cookie ID and optionally cascade delete all related data.
        
        Args:
            user_id: MongoDB document ID of the user (optional)
            cookie_id: Cookie ID of the user (optional)
            cascade: If True, also delete all sessions (and their messages/turns/summaries)
            
        Returns:
            Dictionary with deletion counts: {
                "user_deleted": bool,
                "sessions_deleted": int,
                "messages_deleted": int,
                "turns_deleted": int,
                "summaries_deleted": int
            }
            
        Raises:
            ValueError: If neither user_id nor cookie_id is provided
        
        Traced as CHAIN span for service-level orchestration.
        """
        if not user_id and not cookie_id:
            raise ValueError("Either user_id or cookie_id must be provided")
        
        return await User.delete_by_id_or_cookie(
            user_id=user_id, 
            cookie_id=cookie_id, 
            cascade=cascade
        )
