from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.db.document_models import get_document_models
from ai_server.types import Feedback
from ai_server.utils.tracing import trace_operation


class MessageService:
    @staticmethod
    def _message_model():
        return get_document_models().message

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def update_message_feedback(cls, client_message_id: str, feedback: Feedback | None, cookie_id: str) -> dict:
        """
        Update the feedback for a message.
        
        Args:
            client_message_id: The frontend-generated message ID (from AI SDK)
            feedback: The feedback value (LIKE, DISLIKE, or None for neutral)
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with update info: {
                "message_updated": bool,
                "message_id": str,
                "feedback": str | None
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        message_model = cls._message_model()
        return await message_model.update_feedback_by_client_id(
            client_message_id,
            feedback,
            client_id=cookie_id,
        )
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_message(cls, client_message_id: str, cookie_id: str) -> dict:
        """
        Delete a message by its client ID.
        
        Args:
            client_message_id: The frontend-generated message ID (from AI SDK)
            cookie_id: The user's cookie ID for authorization
            
        Returns:
            Dictionary with deletion info: {
                "message_deleted": bool,
                "deleted_count": int
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        message_model = cls._message_model()
        return await message_model.delete_by_client_message_id_and_client_id(
            client_message_id,
            client_id=cookie_id,
        )
