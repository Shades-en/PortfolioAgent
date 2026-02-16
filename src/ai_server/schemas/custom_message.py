from __future__ import annotations

from pydantic import Field
from beanie import Link as _BeanieLink

from omniagent.schemas.mongo.message import Message as BaseMessage
from omniagent.exceptions import MessageUpdateError

from ai_server.types import Feedback

# Re-export Link so Pydantic can resolve forward references inherited from BaseMessage
Link = _BeanieLink


class CustomMessage(BaseMessage):
    feedback: Feedback | None = Field(default=None)

    @classmethod
    async def update_feedback(cls, client_message_id: str, feedback: Feedback | None, user_id: str) -> dict:
        try:
            message = await cls.get_by_client_id(client_message_id, user_id)

            if not message:
                raise MessageUpdateError(
                    "Message not found",
                    details=f"client_message_id={client_message_id}",
                )

            message.feedback = feedback
            await message.save()

            return {
                "message_updated": True,
                "message_id": client_message_id,
                "feedback": feedback.value if feedback else None,
            }
        except MessageUpdateError:
            raise
        except Exception as exc:
            raise MessageUpdateError(
                "Failed to update message feedback",
                details=f"client_message_id={client_message_id}, feedback={feedback}, error={str(exc)}",
            )
