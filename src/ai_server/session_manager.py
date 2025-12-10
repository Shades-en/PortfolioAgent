import asyncio
from typing import List, Tuple
from pydantic import ValidationError

from ai_server.types.state import State
from ai_server.types.message import MessageDTO, FunctionCallRequest
from ai_server.schemas import User, Session, Turn, Summary, Role
from ai_server.config import MAX_TURNS_TO_FETCH, MAX_TOKEN_THRESHOLD
from ai_server.api.exceptions.schema_exceptions import MessageParseException


class SessionManager():
    def __init__(
        self, 
        user_id: str | None, 
        session_id: str | None, 
        user_cookie: str,
        turn_number: int,
        new_chat: bool, 
        new_user: bool,
        state: dict = {}, 
    ):
        self.state = self._inititialise_state(
            turn_number=turn_number,
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
        turn_number: int, 
        new_chat: bool, 
        new_user: bool,
        state: dict
    ) -> State:
        state = State(
            turn_number=turn_number,
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
        else:
            # Existing user, existing chat - fetch session only
            if self.session_id:
                self.session = await Session.get_by_id(self.session_id)

    async def _fetch_context(self) -> Tuple[List[Turn], Summary | None]:
        """
        Fetch the latest N turns and the latest summary in parallel.
        
        Returns:
            Tuple of (turns, summary) where turns is a list of Turn objects
            and summary is the latest Summary or None.
        """
        if not self.state.new_chat:
            return [], None
        
        turns_task = Turn.get_latest_by_session(
            session_id=str(self.session.id),
            limit=MAX_TURNS_TO_FETCH
        )
        summary_task = Summary.get_latest_by_session(session_id=str(self.session.id))
        
        turns, summary = await asyncio.gather(turns_task, summary_task)
        return turns, summary

    def update_state(self, **kwargs) -> None:
        """
        Update state with provided key-value pairs.
        
        Args:
            **kwargs: Key-value pairs to update in state.
        """
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def _parse_message_dict_to_message(self, message_dict: dict) -> MessageDTO:
        """
        Parse a message dictionary to a MessageDTO object.
        
        Args:
            message_dict: Dictionary containing message data from DB
            
        Returns:
            MessageDTO object for in-memory use
        """
        try:
            # Parse function_call if present
            function_call = None
            if message_dict.get("function_call"):
                function_call = FunctionCallRequest(**message_dict["function_call"])
            
            return MessageDTO(
                role=message_dict.get("role"),
                tool_call_id=message_dict.get("tool_call_id", "null"),
                metadata=message_dict.get("metadata", {}),
                content=message_dict.get("content"),
                function_call=function_call,
                token_count=message_dict.get("token_count", 0),
                created_at=message_dict.get("created_at"),
            )
        except (ValidationError, ValueError) as e:
            raise MessageParseException(
                message="Failed to parse message dictionary to MessageDTO",
                note=f"message_dict={message_dict}, error={str(e)}"
            )

    def _extract_messages_from_turns(self, turns: List[dict]) -> List[MessageDTO]:
        """
        Extract all messages from turns and convert to MessageDTO objects in chronological order.
        
        Args:
            turns: List of turn dictionaries (already sorted chronologically)
            
        Returns:
            Flat list of MessageDTO objects in chronological order
        """
        messages = []
        for turn in turns:
            message_dicts = turn.get("messages", [])
            for msg_dict in message_dicts:
                messages.append(self._parse_message_dict_to_message(msg_dict))
        return messages

    async def get_context_and_update_state(self) -> Tuple[List[MessageDTO], Summary | None]:
        """
        Fetch context (turns + summary) and user/session in parallel, then update state.
        
        Algorithm:
        1. Fetch turns after summary.end_turn_number and calculate total tokens
        2. If tokens < MAX_TOKEN_THRESHOLD, fetch additional older turns until threshold met
        3. Update state with turns_after_last_summary, total_token_after_last_summary, active_summary
        4. Return messages in chronological order with summary
        
        Returns:
            Tuple of (messages, summary) - messages from selected turns in order of arrival.
        """
        # Fetch context and user/session in parallel
        context_task = self._fetch_context()
        user_or_session_task = self._fetch_user_or_session()
        (all_turns, summary), _ = await asyncio.gather(context_task, user_or_session_task)
        
        if not all_turns:
            return [], summary
        
        # Reverse to get chronological order (DB returns descending)
        all_turns.reverse()
        
        # Determine the start point based on summary
        end_turn_number = summary.end_turn_number if summary else 0
        
        # Step 1: Get turns after summary (end_turn_number + 1 to current)
        turns_after_summary = [t for t in all_turns if t["turn_number"] > end_turn_number]
        turns_after_last_summary = len(turns_after_summary)
        total_token_after_last_summary = sum(t["turn_token_count"] for t in turns_after_summary)
        
        # Step 2: Collect context turns - start with turns after summary
        context_turns = turns_after_summary.copy()
        
        # Step 3 & 4: If tokens < threshold, fetch additional older turns
        if total_token_after_last_summary < MAX_TOKEN_THRESHOLD:
            # Get turns at or before end_turn_number (older turns)
            older_turns = [t for t in all_turns if t["turn_number"] <= end_turn_number]
            # Reverse to process from most recent to oldest
            older_turns.reverse()
            
            for turn in older_turns:
                total_token_after_last_summary += turn["turn_token_count"]
                context_turns.insert(0, turn)  # Insert at beginning to maintain order
                if total_token_after_last_summary >= MAX_TOKEN_THRESHOLD:
                    break
        
        # Update state
        self.update_state(
            turns_after_last_summary=turns_after_last_summary,
            total_token_after_last_summary=total_token_after_last_summary,
            active_summary=summary
        )
        
        # Extract messages from turns
        messages = self._extract_messages_from_turns(context_turns)
        
        return messages, summary
    
    def create_fallback_messages(self, user_query_dto: MessageDTO) -> List[MessageDTO]:
        """
        Create fallback messages with user query and error response.
        
        Args:
            user_query_dto: The original user query message
            
        Returns:
            List containing user query and error response message
        """
        # Create error response DTO
        error_message_dto = MessageDTO(
            role=Role.AI,
            tool_call_id="null",
            metadata={"error_type": "insertion_failure"},
            content="I apologize, but something went wrong while processing your request. Please try again.",
            function_call=None,
            token_count=0,
            error=True
        )
        
        # Return list with user message and error response
        return [user_query_dto, error_message_dto]

    async def update_user_session(self, messages: List[MessageDTO], summary: Summary | None) -> None:
        session = self.session
        user = self.user
        
        # Case 1: New user and new session - create both atomically
        if not session and not user:
            self.session = await Session.create_with_user(cookie_id=self.user_cookie)        
        # Case 2: Existing user, new session - create session for existing user
        elif not session and user:
            self.session = await Session.create_for_existing_user(user=user)
        
        # Insert messages and turn for the session
        if self.session:
            turn_number = self.state.turn_number
            try:
                await self.session.insert_messages_and_turn(
                    turn_number=turn_number,
                    messages=messages,
                    summary=summary
                )
            except Exception as e:
                # If insertion fails, still save user message and error response
                if not messages:
                    return
                
                # Create fallback messages with error response
                fallback_messages = self._create_fallback_messages(messages[0])
                
                await self.session.insert_messages_and_turn(
                    turn_number=turn_number,
                    messages=fallback_messages,
                    summary=summary
                )

    
        
        