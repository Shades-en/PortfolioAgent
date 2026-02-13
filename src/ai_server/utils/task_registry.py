"""
Task registry for managing active streaming tasks.

Allows cancellation of in-progress chat streams by session_id.
"""
import asyncio
from typing import Dict

import logging

logger = logging.getLogger(__name__)

# In-memory task registry (works for single-process deployments)
_active_tasks: Dict[str, asyncio.Task] = {}


def register_task(session_id: str, task: asyncio.Task) -> None:
    """Register a task for a session, allowing it to be cancelled later."""
    _active_tasks[session_id] = task
    logger.debug(f"Registered task for session {session_id}")


def cancel_task(session_id: str) -> bool:
    """
    Cancel a task by session_id.
    
    Returns:
        True if task was found and cancelled, False otherwise.
    """
    task = _active_tasks.get(session_id)
    if task and not task.done():
        task.cancel()
        logger.info(f"Cancelled task for session {session_id}")
        return True
    logger.debug(f"No active task found for session {session_id}")
    return False


def unregister_task(session_id: str) -> None:
    """Remove a task from the registry after completion."""
    removed = _active_tasks.pop(session_id, None)
    if removed:
        logger.debug(f"Unregistered task for session {session_id}")
