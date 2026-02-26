from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.persistence import get_context

from ai_server.utils.tracing import trace_operation


class UserService:
    @staticmethod
    def _user_repo():
        return get_context().repositories.users

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_user(cls, cookie_id: str):
        return await cls._user_repo().get_by_client_id(client_id=cookie_id)

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_user(
        cls,
        cookie_id: str,
        cascade: bool = True,
    ) -> dict:
        return await cls._user_repo().delete_by_client_id(
            client_id=cookie_id,
            cascade=cascade,
        )

