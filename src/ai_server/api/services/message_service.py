from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.persistence import get_context
from ai_server.persistence.extensions import MessageRepositoryWithFeedbackProtocol
from ai_server.types import Feedback
from ai_server.utils.tracing import trace_operation


class MessageService:
    @staticmethod
    def _message_repo() -> MessageRepositoryWithFeedbackProtocol:
        repository = get_context().repositories.messages
        if not isinstance(repository, MessageRepositoryWithFeedbackProtocol):
            raise RuntimeError(
                "Message repository is not initialized with required feedback behavior."
            )
        return repository

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def update_message_feedback(cls, client_message_id: str, feedback: Feedback | None, cookie_id: str) -> dict:
        return await cls._message_repo().update_feedback_by_client_id(
            client_message_id=client_message_id,
            feedback=feedback,
            client_id=cookie_id,
        )

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_message(cls, client_message_id: str, cookie_id: str) -> dict:
        return await cls._message_repo().delete_by_client_message_id_and_client_id(
            client_message_id=client_message_id,
            client_id=cookie_id,
        )
