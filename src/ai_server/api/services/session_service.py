import math

from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry.trace import SpanKind

from omniagent.config import DEFAULT_MESSAGE_PAGE_SIZE, DEFAULT_SESSION_PAGE_SIZE
from omniagent.persistence import get_context
from ai_server.persistence.extensions import SessionRepositoryWithFavoritesProtocol
from ai_server.utils.tracing import trace_operation


class SessionService:
    @staticmethod
    def _session_repo() -> SessionRepositoryWithFavoritesProtocol:
        repository = get_context().repositories.sessions
        if not isinstance(repository, SessionRepositoryWithFavoritesProtocol):
            raise RuntimeError(
                "Session repository is not initialized with required favorites behavior."
            )
        return repository

    @staticmethod
    def _message_repo():
        return get_context().repositories.messages

    @staticmethod
    def _session_manager_cls():
        return get_context().session_manager_cls

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session(cls, session_id: str, cookie_id: str) -> dict | None:
        session = await cls._session_repo().get_by_id_and_client_id(
            session_id=session_id,
            client_id=cookie_id,
        )
        if session is None:
            return None
        return dict(session.data)

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_session_messages(
        cls,
        session_id: str,
        cookie_id: str,
        page: int = 1,
        page_size: int = DEFAULT_MESSAGE_PAGE_SIZE,
    ) -> dict:
        session = await cls._session_repo().get_by_id_and_client_id(
            session_id=session_id,
            client_id=cookie_id,
        )
        if session is None:
            return {
                "count": 0,
                "total_count": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "has_next": False,
                "has_previous": False,
                "results": [],
            }

        message_page = await cls._message_repo().list_by_session(
            session_id=session_id,
            page=page,
            page_size=page_size,
        )
        total_pages = math.ceil(message_page.total_count / page_size) if message_page.total_count > 0 else 0
        return {
            "count": len(message_page.items),
            "total_count": message_page.total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
            "results": [message.data for message in message_page.items],
        }

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_all_session_messages(cls, session_id: str, cookie_id: str) -> dict:
        session = await cls._session_repo().get_by_id_and_client_id(
            session_id=session_id,
            client_id=cookie_id,
        )
        if session is None:
            return {"count": 0, "results": []}

        messages = await cls._message_repo().list_all_by_session(session_id=session_id)
        return {
            "count": len(messages),
            "results": [message.data for message in messages],
        }

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_user_sessions(
        cls,
        cookie_id: str,
        page: int = 1,
        page_size: int = DEFAULT_SESSION_PAGE_SIZE,
    ) -> dict:
        session_page = await cls._session_repo().list_by_client_id(
            client_id=cookie_id,
            page=page,
            page_size=page_size,
        )
        total_pages = math.ceil(session_page.total_count / page_size) if session_page.total_count > 0 else 0
        return {
            "count": len(session_page.items),
            "total_count": session_page.total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_previous": page > 1,
            "results": [session.data for session in session_page.items],
        }

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_all_user_sessions(cls, cookie_id: str) -> dict:
        sessions = await cls._session_repo().list_all_by_client_id(client_id=cookie_id)
        return {
            "count": len(sessions),
            "results": [session.data for session in sessions],
        }

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def get_starred_user_sessions(cls, cookie_id: str) -> dict:
        sessions = await cls._session_repo().list_starred_by_client_id(client_id=cookie_id)
        return {
            "count": len(sessions),
            "results": [session.data for session in sessions],
        }

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def update_session_starred(cls, session_id: str, starred: bool, cookie_id: str) -> dict:
        return await cls._session_repo().update_starred_by_client_id(
            session_id=session_id,
            starred=starred,
            client_id=cookie_id,
        )

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def rename_session(cls, session_id: str, name: str, cookie_id: str) -> dict:
        return await cls._session_repo().rename_by_client_id(
            session_id=session_id,
            name=name,
            client_id=cookie_id,
        )

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_session(cls, session_id: str, cookie_id: str) -> dict:
        return await cls._session_repo().delete_with_related_by_client_id(
            session_id=session_id,
            client_id=cookie_id,
        )

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def delete_all_user_sessions(cls, cookie_id: str) -> dict:
        return await cls._session_repo().delete_all_by_client_id(client_id=cookie_id)

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=OpenInferenceSpanKindValues.CHAIN)
    async def generate_chat_name(
        cls,
        query: str,
        turns_between_chat_name: int,
        max_chat_name_length: int,
        max_chat_name_words: int,
        session_id: str | None = None,
        cookie_id: str | None = None,
    ) -> dict:
        session_manager_cls = cls._session_manager_cls()
        chat_name = await session_manager_cls.generate_chat_name(
            query=query,
            turns_between_chat_name=turns_between_chat_name,
            max_chat_name_length=max_chat_name_length,
            max_chat_name_words=max_chat_name_words,
            session_id=session_id,
            client_id=cookie_id,
        )
        return {"name": chat_name, "session_id": session_id}
