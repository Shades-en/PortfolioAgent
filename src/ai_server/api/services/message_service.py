from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from ai_server.schemas import Message, Feedback
from ai_server.utils.tracing import trace_operation


class MessageService:
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def update_message_feedback(cls, message_id: str, feedback: Feedback | None, user_id: str) -> dict:
        """
        Update the feedback for a message.
        
        Args:
            message_id: The message ID to update
            feedback: The feedback value (LIKE, DISLIKE, or None for neutral)
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with update info: {
                "message_updated": bool,
                "message_id": str,
                "feedback": str | None
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Message.update_feedback(message_id, feedback, user_id=user_id)
    
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_message(cls, message_id: str, user_id: str) -> dict:
        """
        Delete a message by its ID.
        
        Args:
            message_id: The message ID to delete
            user_id: The user's MongoDB document ID for authorization
            
        Returns:
            Dictionary with deletion info: {
                "message_deleted": bool,
                "deleted_count": int
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Message.delete_by_id(message_id, user_id=user_id)
