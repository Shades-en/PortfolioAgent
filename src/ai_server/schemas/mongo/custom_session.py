from __future__ import annotations

from typing import List

from opentelemetry.trace import SpanKind
from pydantic import Field

from omniagent.exceptions import SessionRetrievalError, SessionUpdateError
from omniagent.schemas.mongo.session import Session as BaseSession
from omniagent.tracing import CustomSpanKinds, trace_operation


class CustomSession(BaseSession):
    starred: bool = Field(default=False)

    @classmethod
    @trace_operation(kind=SpanKind.INTERNAL, open_inference_kind=CustomSpanKinds.DATABASE.value)
    async def update_starred_by_client_id(cls, session_id: str, starred: bool, client_id: str) -> dict:
        """
        Update the starred status for a session, authorized by client_id.

        Args:
            session_id: The session ID to update
            starred: Whether the session should be starred (True) or unstarred (False)
            client_id: The user's client ID for authorization

        Returns:
            Dictionary with update info: {
                "session_updated": bool,
                "session_id": str,
                "starred": bool
            }

        Raises:
            SessionUpdateError: If update fails
        """
        try:
            session = await cls.get_by_id_and_client_id(session_id, client_id)

            if not session:
                raise SessionUpdateError(
                    "Session not found",
                    details=f"session_id={session_id}",
                )

            session.starred = starred
            await session.save()

            return {
                "session_updated": True,
                "session_id": session_id,
                "starred": starred,
            }
        except SessionUpdateError:
            raise
        except Exception as exc:
            raise SessionUpdateError(
                "Failed to update session starred status",
                details=f"session_id={session_id}, starred={starred}, error={str(exc)}",
            )

    @classmethod
    async def get_starred_by_user_client_id(cls, client_id: str) -> List[CustomSession]:
        """
        Get all starred sessions for a user by client ID, sorted by most recently updated first.
        Uses MongoDB aggregation with $lookup to join with users collection.

        Args:
            client_id: The user's client ID

        Returns:
            List of starred Session documents sorted by most recently updated first

        Raises:
            SessionRetrievalError: If retrieval fails
        """
        try:
            pipeline = [
                {
                    "$lookup": {
                        "from": "users",
                        "localField": "user.$id",
                        "foreignField": "_id",
                        "as": "user_data",
                    }
                },
                {
                    "$unwind": "$user_data",
                },
                {
                    "$match": {
                        "user_data.client_id": client_id,
                        "starred": True,
                    }
                },
                {
                    "$sort": {
                        "updated_at": -1,
                    }
                },
            ]

            sessions = await cls.aggregate(pipeline).to_list()
            return [cls.model_validate(session) for session in sessions]
        except Exception as exc:
            raise SessionRetrievalError(
                "Failed to retrieve starred sessions for user by client_id",
                details=f"client_id={client_id}, error={str(exc)}",
            )
