from __future__ import annotations

from sqlalchemy import Boolean, select
from sqlalchemy.orm import Mapped, mapped_column

from omniagent.db.postgres.engine import get_sessionmaker
from omniagent.exceptions import SessionRetrievalError, SessionUpdateError
from omniagent.schemas.postgres.models import Session as BaseSession
from omniagent.schemas.postgres.models import User


class CustomSession(BaseSession):
    starred: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    @classmethod
    async def list_starred_by_client_id(cls, client_id: str) -> list[CustomSession]:
        try:
            sessionmaker = get_sessionmaker()
            async with sessionmaker() as session:
                result = await session.execute(
                    select(cls)
                    .join(User, cls.user_id == User.id)
                    .where(User.client_id == client_id, cls.starred.is_(True))
                    .order_by(cls.updated_at.desc())
                )
                return list(result.scalars().all())
        except Exception as exc:
            raise SessionRetrievalError(
                "Failed to retrieve starred sessions for user by client_id",
                details=f"client_id={client_id}, error={exc}",
            ) from exc

    @classmethod
    async def update_starred_by_client_id(cls, session_id: str, starred: bool, client_id: str) -> dict:
        sessionmaker = get_sessionmaker()
        async with sessionmaker.begin() as session:
            try:
                row = (
                    await session.execute(
                        select(cls)
                        .join(User, cls.user_id == User.id)
                        .where(cls.id == session_id, User.client_id == client_id)
                    )
                ).scalar_one_or_none()

                if row is None:
                    return {
                        "session_updated": False,
                        "session_id": session_id,
                        "starred": starred,
                    }

                row.starred = starred
                return {
                    "session_updated": True,
                    "session_id": session_id,
                    "starred": starred,
                }
            except Exception as exc:
                raise SessionUpdateError(
                    "Failed to update session starred status",
                    details=f"session_id={session_id}, starred={starred}, client_id={client_id}, error={exc}",
                ) from exc
