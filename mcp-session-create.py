from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import asyncio
import uuid

# Initialize FastMCP server

mcp = FastMCP(“Session Manager”)

# Session storage

sessions: Dict[str, Dict[str, Any]] = {}
SESSION_DURATION = timedelta(minutes=60)

class SessionManager:
“”“Manages user sessions with automatic expiration”””

```
@staticmethod
def create_session(user_id: str, data: Dict[str, Any] = None) -> str:
    """Create a new session for a user"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "user_id": user_id,
        "created_at": datetime.now(),
        "expires_at": datetime.now() + SESSION_DURATION,
        "data": data or {}
    }
    return session_id

@staticmethod
def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session if it exists and is valid"""
    if session_id not in sessions:
        return None
    
    session = sessions[session_id]
    
    # Check if session has expired
    if datetime.now() > session["expires_at"]:
        SessionManager.remove_session(session_id)
        return None
    
    return session

@staticmethod
def update_session(session_id: str, data: Dict[str, Any]) -> bool:
    """Update session data"""
    session = SessionManager.get_session(session_id)
    if session:
        session["data"].update(data)
        return True
    return False

@staticmethod
def remove_session(session_id: str) -> bool:
    """Remove a session"""
    if session_id in sessions:
        del sessions[session_id]
        return True
    return False

@staticmethod
def cleanup_expired_sessions():
    """Remove all expired sessions"""
    now = datetime.now()
    expired = [
        sid for sid, session in sessions.items()
        if now > session["expires_at"]
    ]
    for sid in expired:
        del sessions[sid]
    return len(expired)
```

# Background task to cleanup expired sessions

async def session_cleanup_task():
“”“Background task that runs every 5 minutes to clean up expired sessions”””
while True:
await asyncio.sleep(300)  # 5 minutes
count = SessionManager.cleanup_expired_sessions()
if count > 0:
print(f”Cleaned up {count} expired sessions”)

# MCP Tools

@mcp.tool()
def create_user_session(user_id: str, initial_data: dict = None) -> dict:
“””
Create a new session for a user

```
Args:
    user_id: Unique identifier for the user
    initial_data: Optional initial session data

Returns:
    Dictionary with session_id and expiration info
"""
session_id = SessionManager.create_session(user_id, initial_data)
session = sessions[session_id]

return {
    "session_id": session_id,
    "user_id": user_id,
    "expires_at": session["expires_at"].isoformat(),
    "expires_in_minutes": 60
}
```

@mcp.tool()
def get_user_session(session_id: str) -> dict:
“””
Get session data if valid

```
Args:
    session_id: The session identifier

Returns:
    Session data or error message
"""
session = SessionManager.get_session(session_id)

if not session:
    return {"error": "Session not found or expired"}

time_remaining = session["expires_at"] - datetime.now()

return {
    "session_id": session_id,
    "user_id": session["user_id"],
    "data": session["data"],
    "created_at": session["created_at"].isoformat(),
    "expires_at": session["expires_at"].isoformat(),
    "minutes_remaining": int(time_remaining.total_seconds() / 60)
}
```

@mcp.tool()
def update_user_session(session_id: str, data: dict) -> dict:
“””
Update session data

```
Args:
    session_id: The session identifier
    data: Data to merge into session

Returns:
    Success status
"""
success = SessionManager.update_session(session_id, data)

if success:
    return {"success": True, "message": "Session updated"}
else:
    return {"success": False, "error": "Session not found or expired"}
```

@mcp.tool()
def delete_user_session(session_id: str) -> dict:
“””
Delete a session

```
Args:
    session_id: The session identifier

Returns:
    Success status
"""
success = SessionManager.remove_session(session_id)

if success:
    return {"success": True, "message": "Session deleted"}
else:
    return {"success": False, "error": "Session not found"}
```

@mcp.tool()
def list_active_sessions() -> dict:
“””
List all active sessions

```
Returns:
    Dictionary with active session info
"""
# Clean up expired sessions first
SessionManager.cleanup_expired_sessions()

active = []
for sid, session in sessions.items():
    time_remaining = session["expires_at"] - datetime.now()
    active.append({
        "session_id": sid,
        "user_id": session["user_id"],
        "created_at": session["created_at"].isoformat(),
        "minutes_remaining": int(time_remaining.total_seconds() / 60)
    })

return {
    "total_sessions": len(active),
    "sessions": active
}
```

@mcp.tool()
def cleanup_expired() -> dict:
“””
Manually trigger cleanup of expired sessions

```
Returns:
    Number of sessions cleaned up
"""
count = SessionManager.cleanup_expired_sessions()
return {
    "cleaned_up": count,
    "active_sessions": len(sessions)
}
```

# Start the cleanup task when server starts

@mcp.lifespan()
async def lifespan():
“”“Lifespan context manager for background tasks”””
# Start cleanup task
cleanup_task = asyncio.create_task(session_cleanup_task())

```
yield

# Cleanup on shutdown
cleanup_task.cancel()
try:
    await cleanup_task
except asyncio.CancelledError:
    pass
```

if **name** == “**main**”:
# Run the server
mcp.run()