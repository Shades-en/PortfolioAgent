from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.schemas import User
from ai_server.utils.tracing import trace_operation


class UserService:
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_user(cls, cookie_id: str) -> User | None:
        """
        Get a user by their cookie ID.
        
        Args:
            cookie_id: Cookie ID of the user
            
        Returns:
            User document if found, None otherwise
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await User.get_by_client_id(client_id=cookie_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_user(
        cls, 
        cookie_id: str,
        cascade: bool = True
    ) -> dict:
        """
        Delete a user by their cookie ID and optionally cascade delete all related data.
        
        Args:
            cookie_id: Cookie ID of the user
            cascade: If True, also delete all sessions (and their messages/turns/summaries)
            
        Returns:
            Dictionary with deletion counts: {
                "user_deleted": bool,
                "sessions_deleted": int,
                "messages_deleted": int,
                "turns_deleted": int,
                "summaries_deleted": int
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await User.delete_by_client_id(
            client_id=cookie_id, 
            cascade=cascade
        )
