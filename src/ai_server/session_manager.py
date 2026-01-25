import asyncio
from typing import List, Tuple

import logging

from ai_server.types.state import State
from ai_server.types.message import MessageDTO
from ai_server.schemas import User, Session, Message, Summary
from ai_server.config import MAX_TURNS_TO_FETCH, MAX_TOKEN_THRESHOLD, MONGODB_OBJECTID_LENGTH
from ai_server.api.exceptions.db_exceptions import (
    SessionNotFoundException,
    UserNotFoundException,
)
from ai_server.utils.general import generate_id
from ai_server.utils.tracing import trace_method, track_state_change, CustomSpanKinds

logger = logging.getLogger(__name__)

class SessionManager():
    def __init__(
        self, 
        user_id: str | None, 
        session_id: str | None, 
        user_cookie: str,
        new_chat: bool, 
        new_user: bool,
        state: dict = {}, 
    ):
        self.state = self._inititialise_state(
            new_chat=new_chat,
            new_user=new_user,
            state=state
        )
        self.user_id = user_id
        self.session_id = session_id
        self.user_cookie = user_cookie
        self.user: User | None = None
        self.session: Session | None = None

    def _inititialise_state(
        self, 
        new_chat: bool, 
        new_user: bool,
        state: dict
    ) -> State:
        state = State(
            new_chat=new_chat,
            new_user=new_user,
            user_defined_state=state
        )
        return state
    
    async def _fetch_user_or_session(self) -> None:
        """
        Fetch user or session based on new_user and new_chat flags.
        
        - new_user=False, new_chat=True  -> Fetch user only
        - new_user=False, new_chat=False -> Fetch session only
        - new_user=True -> Do nothing (new user, no data to fetch)
        
        """
        if self.state.new_user:
            # New user - nothing to fetch
            return
        
        if self.state.new_chat:
            # Existing user, new chat - fetch user only
            self.user = await User.get_by_id_or_cookie(self.user_id, self.user_cookie)
            if not self.user:
                raise UserNotFoundException(
                    message="User not found for provided identifiers",
                    note=f"user_id={self.user_id}, user_cookie={self.user_cookie}"
                )
        else:
            # Existing user, existing chat - fetch session only s
            if self.session_id:
                self.session = await Session.get_by_id(self.session_id)
            if not self.session:
                raise SessionNotFoundException(
                    message=f"Session not found for session_id: {self.session_id}",
                    note=f"session_id={self.session_id}"
                )

    @trace_method(
        kind=CustomSpanKinds.DATABASE.value,
        graph_node_id="fetch_context",
        capture_input=False,
        capture_output=False
    )
    async def _fetch_context(self) -> Tuple[List[Message], Summary | None]:
        """
        Fetch the latest N turns (as messages) and the latest summary in parallel.
        
        Returns:
            Tuple of (messages, summary) where messages is a list of Message objects
            from the latest turns and summary is the latest Summary or None.
        
        Traced as DATABASE span for database fetch operations.
        """
        if self.state.new_chat or not self.session_id:
            return [], None
        
        messages_task = Message.get_latest_by_session(
            session_id=str(self.session_id),
            current_turn_number=self.state.turn_number,
            max_turns=MAX_TURNS_TO_FETCH,
        )
        summary_task = Summary.get_latest_by_session(session_id=str(self.session_id))
        
        messages, summary = await asyncio.gather(messages_task, summary_task)
        return messages, summary

    def update_state(self, **kwargs) -> None:
        """
        Update state with provided key-value pairs.
        
        Automatically tracks state changes in the current span for observability.
        
        Args:
            **kwargs: Key-value pairs to update in state.
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                old_value = getattr(self.state, key)
                setattr(self.state, key, value)
                
                # Track state change in current span
                track_state_change(key, old_value, value)

    def _convert_messages_to_dtos(self, messages: List[Message]) -> List[MessageDTO]:
        """
        Convert Message documents to MessageDTO objects.
        
        Args:
            messages: List of Message documents (already sorted chronologically)
            
        Returns:
            List of MessageDTO objects in chronological order
        """
        return [
            MessageDTO(
                id=str(msg.id),
                role=msg.role,
                parts=msg.parts,
                metadata=msg.metadata,
                created_at=msg.created_at
            )
            for msg in messages
        ]

    @trace_method(
        kind=CustomSpanKinds.DATABASE.value,
        graph_node_id="session_context_loader",
        capture_input=False,
        capture_output=False
    )
    async def get_context_and_update_state(self) -> Tuple[List[MessageDTO], Summary | None]:
        """
        Fetch context (messages + summary) and user/session in parallel, then update state.
        
        Algorithm:
        1. Fetch messages from latest turns and calculate tokens after summary
        2. If tokens < MAX_TOKEN_THRESHOLD, include additional older messages until threshold met
        3. Update state with turns_after_last_summary, total_token_after_last_summary, active_summary
        4. Return messages in chronological order with summary
        
        Returns:
            Tuple of (messages, summary) - messages from selected turns in order of arrival.
        
        Traced as INTERNAL span for database/session operations.
        """
        # Fetch context and user/session in parallel
        context_task = self._fetch_context()
        user_or_session_task = self._fetch_user_or_session()
        (all_messages, summary), _ = await asyncio.gather(context_task, user_or_session_task)
        if self.session:
            self.update_state(turn_number=self.session.latest_turn_number+1)
        
        if not all_messages:
            return [], summary
        
        # Messages are already in chronological order from get_latest_by_session
        
        # Determine the start point based on summary
        end_turn_number = summary.end_turn_number if summary else 0
        
        # Step 1: Get messages after summary (end_turn_number + 1 to current)
        messages_after_summary = [m for m in all_messages if m.turn_number > end_turn_number]
        
        # Count unique turns after summary
        turns_after_summary = {m.turn_number for m in messages_after_summary}
        turns_after_last_summary = len(turns_after_summary)
        total_token_after_last_summary = sum(m.token_count for m in messages_after_summary)
        
        # Step 2: Collect context messages - start with messages after summary
        context_messages = messages_after_summary.copy()
        
        # Step 3 & 4: If tokens < threshold, fetch additional older messages
        if total_token_after_last_summary < MAX_TOKEN_THRESHOLD:
            # Get messages at or before end_turn_number (older messages)
            older_messages = [m for m in all_messages if m.turn_number <= end_turn_number]
            # Reverse to process from most recent to oldest
            older_messages.reverse()
            
            for message in older_messages:
                total_token_after_last_summary += message.token_count
                context_messages.insert(0, message)  # Insert at beginning to maintain order
                if total_token_after_last_summary >= MAX_TOKEN_THRESHOLD:
                    break
        
        # Update state
        self.update_state(
            turns_after_last_summary=turns_after_last_summary,
            total_token_after_last_summary=total_token_after_last_summary,
            active_summary=summary
        )
        
        # Convert Message documents to MessageDTOs
        message_dtos = self._convert_messages_to_dtos(context_messages)
        
        return message_dtos, summary
    
    def create_fallback_messages(self, user_query_dto: MessageDTO) -> List[MessageDTO]:
        """
        Create fallback messages with user query and error response.
        
        Args:
            user_query_dto: The original user query message
            
        Returns:
            List containing user query and error response message
        """
        # Create error response DTO using new schema
        error_message_dto = MessageDTO.create_ai_message(
            message_id=generate_id(MONGODB_OBJECTID_LENGTH),
            metadata={"error_type": "insertion_failure"}
        ).update_ai_text_message(
            text="I apologize, but something went wrong while processing your request. Please try again."
        )
        
        # Return list with user message and error response
        return [user_query_dto, error_message_dto]

    @trace_method(
        kind=CustomSpanKinds.DATABASE.value,
        graph_node_id="session_updater",
        capture_input=False,
        capture_output=False
    )
    async def update_user_session(
        self, 
        messages: List[MessageDTO], 
        summary: Summary | None, 
        chat_name: str | None,
        regenerated_summary: bool
    ) -> List[MessageDTO]:
        # Track if session existed before this method (for parallel name update logic)
        session_existed = self.session is not None
        
        # Case 1: New user and new session - create both atomically
        if not self.session and not self.user:
            if chat_name:
                self.session = await Session.create_with_user(
                    cookie_id=self.user_cookie, 
                    session_name=chat_name
                )
            else:
                self.session = await Session.create_with_user(cookie_id=self.user_cookie)
        # Case 2: Existing user, new session - create session for existing user
        elif not self.session and self.user:
            if chat_name:
                self.session = await Session.create_for_existing_user(
                    user=self.user, 
                    session_name=chat_name
                )
            else:
                self.session = await Session.create_for_existing_user(user=self.user)
        
        # Insert messages for the session
        if self.session:
            turn_number = self.state.turn_number
            try:
                if regenerated_summary:
                    await Summary.create_with_session(
                        session=self.session,
                        summary=summary
                    )
                # Ensure writes happen sequentially to avoid Mongo write conflicts
                await self.session.insert_messages(
                    messages=messages,
                    turn_number=turn_number,
                    previous_summary=summary,
                )

                if session_existed and chat_name:
                    await self.session.update_name(chat_name)
                
            except Exception as e:
                logger.error(f"Failed to insert messages for session {self.session_id}: {str(e)}")
                # If insertion fails, still save user message and error response
                if not messages:
                    return
                
                # Create fallback messages with error response
                messages = self.create_fallback_messages(messages[0])
                
                await self.session.insert_messages(
                    messages=messages,
                    turn_number=turn_number,
                    previous_summary=summary,
                )

        return messages
    
        
        