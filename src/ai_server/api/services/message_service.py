from openinference.semconv.trace import OpenInferenceSpanKindValues

from ai_server.schemas import Message
from ai_server.utils.tracing import trace_operation


class MessageService:
    @trace_operation(kind=OpenInferenceSpanKindValues.CHAIN)
    @classmethod
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
