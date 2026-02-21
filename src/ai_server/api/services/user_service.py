from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.db.document_models import get_document_models
from ai_server.utils.tracing import trace_operation


class UserService:
    @staticmethod
    def _user_model():
        return get_document_models().user

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_user(cls, cookie_id: str):
        """
        Get a user by their cookie ID.
        
        Args:
            cookie_id: Cookie ID of the user
            
        Returns:
            User document if found, None otherwise
        
        Traced as CHAIN span for service-level orchestration.
        """
        user_model = cls._user_model()
        return await user_model.get_by_client_id(client_id=cookie_id)
    
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
        user_model = cls._user_model()
        return await user_model.delete_by_client_id(
            client_id=cookie_id, 
            cascade=cascade
        )
