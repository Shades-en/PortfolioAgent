from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from ai_server.schemas import Message
from ai_server.utils.tracing import trace_operation


class MessageService:
    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_message(cls, message_id: str) -> dict:
        """
        Delete a message by its ID and remove its reference from any Turn.
        
        Args:
            message_id: The message ID to delete
            
        Returns:
            Dictionary with deletion info: {
                "message_deleted": bool,
                "turns_updated": int
            }
        
        Traced as CHAIN span for service-level orchestration.
        """
        return await Message.delete_by_id(message_id)
