from __future__ import annotations

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from omniagent.db.postgres.engine import get_sessionmaker
from omniagent.exceptions import MessageUpdateError
from omniagent.schemas.postgres.models import Message as BaseMessage
from omniagent.schemas.postgres.models import Session, User

from ai_server.types import Feedback


class CustomMessage(BaseMessage):
    feedback: Mapped[str | None] = mapped_column(String(32), nullable=True, default=None)

    def to_public_dict(self, *, exclude: set[str] | None = None) -> dict:
        payload = super().to_public_dict(exclude=exclude)
        payload["feedback"] = self.feedback
        return payload

    @classmethod
    async def update_feedback_by_client_id(
        cls,
        client_message_id: str,
        feedback: Feedback | None,
        client_id: str,
    ) -> dict:
        sessionmaker = get_sessionmaker()
        async with sessionmaker.begin() as session:
            try:
                row = (
                    await session.execute(
                        select(cls)
                        .join(Session, cls.session_id == Session.id)
                        .join(User, Session.user_id == User.id)
                        .where(
                            cls.client_message_id == client_message_id,
                            User.client_id == client_id,
                        )
                    )
                ).scalar_one_or_none()

                feedback_value = feedback.value if feedback else None
                if row is None:
                    return {
                        "message_updated": False,
                        "message_id": client_message_id,
                        "feedback": feedback_value,
                    }

                row.feedback = feedback_value
                return {
                    "message_updated": True,
                    "message_id": client_message_id,
                    "feedback": feedback_value,
                }
            except Exception as exc:
                raise MessageUpdateError(
                    "Failed to update message feedback",
                    details=f"client_message_id={client_message_id}, client_id={client_id}, feedback={feedback}, error={exc}",
                ) from exc
