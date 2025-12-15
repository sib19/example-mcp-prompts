# “””
FastMCP Multi-Tool Session Management with Shareable User Tokens

Complete implementation of session management with user authentication tokens,
cross-tool state sharing, and secure token-based access control.
“””

from fastmcp import FastMCP, Context
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import asyncio
import secrets
import hashlib
import json

mcp = FastMCP(“Multi-Tool Token Session Manager”)

# ============================================

# 1. TOKEN & SESSION MODELS

# ============================================

@dataclass
class UserToken:
“”“Represents a user authentication token”””
token: str
user_id: str
created_at: datetime
expires_at: datetime
permissions: Set[str] = field(default_factory=set)
metadata: Dict[str, Any] = field(default_factory=dict)

```
def is_valid(self) -> bool:
    """Check if token is still valid"""
    return datetime.now() < self.expires_at

def has_permission(self, permission: str) -> bool:
    """Check if token has specific permission"""
    return permission in self.permissions or "admin" in self.permissions
```

@dataclass
class UserSession:
“”“Represents a user session with shared state”””
session_id: str
user_id: str
token: str
created_at: datetime
last_access: datetime
data: Dict[str, Any] = field(default_factory=dict)
tool_history: List[Dict] = field(default_factory=list)
shared_context: Dict[str, Any] = field(default_factory=dict)

```
def touch(self):
    """Update last access time"""
    self.last_access = datetime.now()
```

# ============================================

# 2. SESSION & TOKEN MANAGER

# ============================================

class SessionTokenManager:
“””
Manages user sessions, tokens, and cross-tool state sharing
“””

```
def __init__(self):
    self.sessions: Dict[str, UserSession] = {}
    self.tokens: Dict[str, UserToken] = {}
    self.user_sessions: Dict[str, List[str]] = {}  # user_id -> session_ids
    self.lock = asyncio.Lock()
    self.token_expiry_hours = 24
    self.session_timeout_minutes = 60

def generate_token(self, length: int = 32) -> str:
    """Generate a secure random token"""
    return secrets.token_urlsafe(length)

def hash_token(self, token: str) -> str:
    """Hash token for secure storage"""
    return hashlib.sha256(token.encode()).hexdigest()

async def create_user_token(
    self,
    user_id: str,
    permissions: Optional[Set[str]] = None,
    expires_in_hours: Optional[int] = None
) -> UserToken:
    """Create a new user token"""
    async with self.lock:
        token_str = self.generate_token()
        expires_in = expires_in_hours or self.token_expiry_hours
        
        token = UserToken(
            token=token_str,
            user_id=user_id,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=expires_in),
            permissions=permissions or {"read", "write"}
        )
        
        self.tokens[token_str] = token
        return token

async def validate_token(self, token: str) -> Optional[UserToken]:
    """Validate and return token if valid"""
    async with self.lock:
        user_token = self.tokens.get(token)
        if user_token and user_token.is_valid():
            return user_token
        elif user_token:
            # Token expired, remove it
            del self.tokens[token]
        return None

async def create_session(
    self,
    user_id: str,
    token: str,
    session_id: Optional[str] = None
) -> UserSession:
    """Create a new user session"""
    async with self.lock:
        session_id = session_id or f"sess_{self.generate_token(16)}"
        
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            token=token,
            created_at=datetime.now(),
            last_access=datetime.now()
        )
        
        self.sessions[session_id] = session
        
        # Track user's sessions
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = []
        self.user_sessions[user_id].append(session_id)
        
        return session

async def get_session(self, session_id: str) -> Optional[UserSession]:
    """Get session by ID"""
    async with self.lock:
        session = self.sessions.get(session_id)
        if session:
            # Check session timeout
            if datetime.now() - session.last_access > timedelta(minutes=self.session_timeout_minutes):
                del self.sessions[session_id]
                return None
            session.touch()
        return session

async def get_session_by_token(self, token: str) -> Optional[UserSession]:
    """Get session by token"""
    async with self.lock:
        for session in self.sessions.values():
            if session.token == token:
                session.touch()
                return session
        return None

async def update_session_data(self, session_id: str, key: str, value: Any):
    """Update session data"""
    session = await self.get_session(session_id)
    if session:
        async with self.lock:
            session.data[key] = value

async def get_session_data(self, session_id: str, key: str, default=None):
    """Get session data"""
    session = await self.get_session(session_id)
    if session:
        return session.data.get(key, default)
    return default

async def log_tool_call(self, session_id: str, tool_name: str, params: Dict, result: Any):
    """Log tool call in session history"""
    session = await self.get_session(session_id)
    if session:
        async with self.lock:
            session.tool_history.append({
                "tool": tool_name,
                "params": params,
                "result": str(result)[:200],  # Truncate
                "timestamp": datetime.now().isoformat()
            })

async def share_data_between_sessions(
    self,
    from_session_id: str,
    to_session_id: str,
    key: str,
    value: Any
) -> bool:
    """Share data between two sessions of the same user"""
    from_session = await self.get_session(from_session_id)
    to_session = await self.get_session(to_session_id)
    
    if from_session and to_session and from_session.user_id == to_session.user_id:
        async with self.lock:
            to_session.shared_context[key] = {
                "value": value,
                "from_session": from_session_id,
                "shared_at": datetime.now().isoformat()
            }
        return True
    return False

async def get_user_sessions(self, user_id: str) -> List[UserSession]:
    """Get all active sessions for a user"""
    async with self.lock:
        session_ids = self.user_sessions.get(user_id, [])
        return [
            self.sessions[sid]
            for sid in session_ids
            if sid in self.sessions
        ]

async def revoke_token(self, token: str) -> bool:
    """Revoke a token and close associated sessions"""
    async with self.lock:
        if token in self.tokens:
            # Remove token
            user_token = self.tokens[token]
            del self.tokens[token]
            
            # Close all sessions using this token
            sessions_to_remove = [
                sid for sid, sess in self.sessions.items()
                if sess.token == token
            ]
            for sid in sessions_to_remove:
                del self.sessions[sid]
            
            return True
        return False
```

# Initialize global manager

manager = SessionTokenManager()

# ============================================

# 3. AUTHENTICATION & SESSION TOOLS

# ============================================

@mcp.tool()
async def create_user_session(
user_id: str,
permissions: str = “read,write”
) -> str:
“””
Create a new user session with authentication token

```
Args:
    user_id: Unique user identifier
    permissions: Comma-separated permissions (e.g., "read,write,admin")

Returns:
    JSON with session_id and token
"""
permission_set = set(p.strip() for p in permissions.split(","))

# Create token
token = await manager.create_user_token(user_id, permission_set)

# Create session
session = await manager.create_session(user_id, token.token)

return json.dumps({
    "session_id": session.session_id,
    "token": token.token,
    "user_id": user_id,
    "expires_at": token.expires_at.isoformat(),
    "permissions": list(permission_set)
}, indent=2)
```

@mcp.tool()
async def validate_session(token: str) -> str:
“””
Validate a user token and return session info

```
Args:
    token: User authentication token
"""
user_token = await manager.validate_token(token)

if not user_token:
    return json.dumps({"valid": False, "error": "Invalid or expired token"})

session = await manager.get_session_by_token(token)

return json.dumps({
    "valid": True,
    "user_id": user_token.user_id,
    "session_id": session.session_id if session else None,
    "permissions": list(user_token.permissions),
    "expires_at": user_token.expires_at.isoformat()
}, indent=2)
```

@mcp.tool()
async def revoke_user_token(token: str) -> str:
“””
Revoke a user token and close all associated sessions

```
Args:
    token: Token to revoke
"""
success = await manager.revoke_token(token)

if success:
    return "Token revoked and sessions closed"
return "Token not found"
```

# ============================================

# 4. CROSS-TOOL SESSION DATA SHARING

# ============================================

@mcp.tool()
async def store_session_data(
token: str,
key: str,
value: str,
data_type: str = “string”
) -> str:
“””
Store data in user session (accessible across all tool calls)

```
Args:
    token: User authentication token
    key: Data key
    value: Data value
    data_type: Type of data (string, json, number)
"""
# Validate token
user_token = await manager.validate_token(token)
if not user_token or not user_token.has_permission("write"):
    return "Error: Invalid token or insufficient permissions"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

# Parse value based on type
parsed_value = value
if data_type == "json":
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format"
elif data_type == "number":
    try:
        parsed_value = float(value)
    except ValueError:
        return "Error: Invalid number format"

await manager.update_session_data(session.session_id, key, parsed_value)
await manager.log_tool_call(session.session_id, "store_session_data", 
                            {"key": key, "type": data_type}, "success")

return f"Data stored: {key} = {value} (type: {data_type})"
```

@mcp.tool()
async def retrieve_session_data(token: str, key: str) -> str:
“””
Retrieve data from user session

```
Args:
    token: User authentication token
    key: Data key to retrieve
"""
# Validate token
user_token = await manager.validate_token(token)
if not user_token or not user_token.has_permission("read"):
    return "Error: Invalid token or insufficient permissions"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

value = await manager.get_session_data(session.session_id, key)

await manager.log_tool_call(session.session_id, "retrieve_session_data",
                           {"key": key}, value or "not_found")

if value is None:
    return f"No data found for key: {key}"

if isinstance(value, (dict, list)):
    return json.dumps(value, indent=2)
return str(value)
```

@mcp.tool()
async def list_session_data(token: str) -> str:
“””
List all data keys stored in the session

```
Args:
    token: User authentication token
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

return json.dumps({
    "session_id": session.session_id,
    "user_id": session.user_id,
    "data_keys": list(session.data.keys()),
    "shared_context_keys": list(session.shared_context.keys())
}, indent=2)
```

# ============================================

# 5. MULTI-TOOL WORKFLOW WITH SHARED STATE

# ============================================

@mcp.tool()
async def start_data_pipeline(token: str, pipeline_name: str, config: str) -> str:
“””
Start a multi-step data pipeline (Step 1)

```
Args:
    token: User authentication token
    pipeline_name: Name of the pipeline
    config: JSON configuration for the pipeline
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

try:
    config_data = json.loads(config)
except json.JSONDecodeError:
    return "Error: Invalid JSON configuration"

pipeline_id = f"pipe_{secrets.token_hex(8)}"
pipeline_state = {
    "pipeline_id": pipeline_id,
    "name": pipeline_name,
    "config": config_data,
    "status": "started",
    "steps_completed": [],
    "created_at": datetime.now().isoformat()
}

await manager.update_session_data(session.session_id, 
                                  f"pipeline_{pipeline_id}", pipeline_state)
await manager.update_session_data(session.session_id, 
                                  "active_pipeline_id", pipeline_id)

await manager.log_tool_call(session.session_id, "start_data_pipeline",
                           {"name": pipeline_name}, pipeline_id)

return json.dumps({
    "message": "Pipeline started",
    "pipeline_id": pipeline_id,
    "next_step": "Call process_data_step to process data"
}, indent=2)
```

@mcp.tool()
async def process_data_step(token: str, step_name: str, input_data: str) -> str:
“””
Process a step in the active pipeline (Step 2)

```
Args:
    token: User authentication token
    step_name: Name of this processing step
    input_data: Data to process
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

pipeline_id = await manager.get_session_data(session.session_id, "active_pipeline_id")
if not pipeline_id:
    return "Error: No active pipeline. Start one with start_data_pipeline"

pipeline_state = await manager.get_session_data(session.session_id, 
                                                f"pipeline_{pipeline_id}")

# Simulate processing
result = f"Processed: {input_data[:100]}..."

pipeline_state["steps_completed"].append({
    "step_name": step_name,
    "input": input_data[:100],
    "result": result,
    "timestamp": datetime.now().isoformat()
})

await manager.update_session_data(session.session_id, 
                                  f"pipeline_{pipeline_id}", pipeline_state)

await manager.log_tool_call(session.session_id, "process_data_step",
                           {"step": step_name}, "completed")

return json.dumps({
    "message": "Step completed",
    "step_name": step_name,
    "result": result,
    "steps_completed": len(pipeline_state["steps_completed"]),
    "next_step": "Call finalize_pipeline to complete"
}, indent=2)
```

@mcp.tool()
async def finalize_pipeline(token: str) -> str:
“””
Finalize and get summary of the active pipeline (Step 3)

```
Args:
    token: User authentication token
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

pipeline_id = await manager.get_session_data(session.session_id, "active_pipeline_id")
if not pipeline_id:
    return "Error: No active pipeline"

pipeline_state = await manager.get_session_data(session.session_id, 
                                                f"pipeline_{pipeline_id}")

pipeline_state["status"] = "completed"
pipeline_state["completed_at"] = datetime.now().isoformat()

await manager.update_session_data(session.session_id, 
                                  f"pipeline_{pipeline_id}", pipeline_state)
await manager.update_session_data(session.session_id, "active_pipeline_id", None)

await manager.log_tool_call(session.session_id, "finalize_pipeline",
                           {"pipeline_id": pipeline_id}, "completed")

return json.dumps({
    "message": "Pipeline completed",
    "pipeline_id": pipeline_id,
    "pipeline_name": pipeline_state["name"],
    "total_steps": len(pipeline_state["steps_completed"]),
    "duration": f"{pipeline_state['created_at']} to {pipeline_state['completed_at']}",
    "steps": pipeline_state["steps_completed"]
}, indent=2)
```

# ============================================

# 6. CROSS-SESSION DATA SHARING

# ============================================

@mcp.tool()
async def share_data_to_session(
token: str,
target_session_id: str,
key: str,
value: str
) -> str:
“””
Share data from current session to another session of the same user

```
Args:
    token: User authentication token
    target_session_id: Target session ID to share with
    key: Data key
    value: Data value to share
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

source_session = await manager.get_session_by_token(token)
if not source_session:
    return "Error: No active session found"

success = await manager.share_data_between_sessions(
    source_session.session_id,
    target_session_id,
    key,
    value
)

if success:
    return f"Data shared successfully to session {target_session_id}"
return "Error: Could not share data. Sessions must belong to same user."
```

@mcp.tool()
async def get_shared_data(token: str) -> str:
“””
Get all data shared with this session from other sessions

```
Args:
    token: User authentication token
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

if not session.shared_context:
    return "No shared data in this session"

return json.dumps(session.shared_context, indent=2)
```

@mcp.tool()
async def list_user_sessions(token: str) -> str:
“””
List all active sessions for the current user

```
Args:
    token: User authentication token
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

sessions = await manager.get_user_sessions(user_token.user_id)

session_list = []
for sess in sessions:
    session_list.append({
        "session_id": sess.session_id,
        "created_at": sess.created_at.isoformat(),
        "last_access": sess.last_access.isoformat(),
        "data_keys": list(sess.data.keys()),
        "tool_calls": len(sess.tool_history)
    })

return json.dumps({
    "user_id": user_token.user_id,
    "total_sessions": len(session_list),
    "sessions": session_list
}, indent=2)
```

# ============================================

# 7. SESSION ANALYTICS & HISTORY

# ============================================

@mcp.tool()
async def get_session_history(token: str, limit: int = 10) -> str:
“””
Get tool call history for the current session

```
Args:
    token: User authentication token
    limit: Maximum number of history items to return
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

history = session.tool_history[-limit:]

return json.dumps({
    "session_id": session.session_id,
    "total_calls": len(session.tool_history),
    "recent_calls": history
}, indent=2)
```

@mcp.tool()
async def get_session_metrics(token: str) -> str:
“””
Get comprehensive metrics for the current session

```
Args:
    token: User authentication token
"""
user_token = await manager.validate_token(token)
if not user_token:
    return "Error: Invalid token"

session = await manager.get_session_by_token(token)
if not session:
    return "Error: No active session found"

duration = datetime.now() - session.created_at

# Count tools used
tool_counts = {}
for call in session.tool_history:
    tool = call["tool"]
    tool_counts[tool] = tool_counts.get(tool, 0) + 1

return json.dumps({
    "session_id": session.session_id,
    "user_id": session.user_id,
    "created_at": session.created_at.isoformat(),
    "last_access": session.last_access.isoformat(),
    "duration_seconds": duration.total_seconds(),
    "total_tool_calls": len(session.tool_history),
    "unique_tools_used": len(tool_counts),
    "tool_usage": tool_counts,
    "stored_data_keys": len(session.data),
    "shared_context_items": len(session.shared_context)
}, indent=2)
```

# ============================================

# 8. USAGE EXAMPLE & DOCUMENTATION

# ============================================

# “””
COMPLETE USAGE FLOW:

## Step 1: Create Session and Get Token

result = await create_user_session(
user_id=“alice@example.com”,
permissions=“read,write,admin”
)

# Returns: {“session_id”: “sess_…”, “token”: “abc123…”}

## Step 2: Use Token Across Multiple Tool Calls

token = “abc123…”  # Save this token!

# Store data (accessible across all tools)

await store_session_data(token, “user_preference”, “dark_mode”, “string”)
await store_session_data(token, “cart”, ‘[“item1”, “item2”]’, “json”)

# Start multi-step workflow

await start_data_pipeline(token, “ETL Process”, ‘{“source”: “db”}’)
await process_data_step(token, “extract”, “data from source”)
await process_data_step(token, “transform”, “processed data”)
await finalize_pipeline(token)

# Retrieve data anytime

await retrieve_session_data(token, “user_preference”)
await list_session_data(token)

## Step 3: Share Data Between Sessions

# User opens new browser tab/device - creates new session

result2 = await create_user_session(“alice@example.com”, “read,write”)
token2 = json.loads(result2)[“token”]

# Share data from first session to second

await share_data_to_session(token, session2_id, “cart”, ‘[“item1”]’)

# Access shared data in second session

await get_shared_data(token2)

## Step 4: View Analytics

await get_session_history(token, limit=20)
await get_session_metrics(token)
await list_user_sessions(token)

## Step 5: Cleanup

await revoke_user_token(token)  # Closes all sessions

# CLIENT IMPLEMENTATION:

# Python Client Example

import requests
import json

class MCPClient:
def **init**(self, server_url):
self.server_url = server_url
self.token = None
self.session_id = None

```
def authenticate(self, user_id, permissions="read,write"):
    response = self.call_tool("create_user_session", {
        "user_id": user_id,
        "permissions": permissions
    })
    data = json.loads(response)
    self.token = data["token"]
    self.session_id = data["session_id"]
    return data

def call_tool(self, tool_name, params):
    # All tools automatically use the stored token
    if self.token and "token" not in params:
        params["token"] = self.token
    
    response = requests.post(
        f"{self.server_url}/tools/{tool_name}",
        json=params
    )
    return response.json()

def store_data(self, key, value, data_type="string"):
    return self.call_tool("store_session_data", {
        "key": key,
        "value": value,
        "data_type": data_type
    })

def get_data(self, key):
    return self.call_tool("retrieve_session_data", {"key": key})
```

# Usage

client = MCPClient(“http://localhost:8000”)
client.authenticate(“alice@example.com”)

# All subsequent calls use the same session/token

client.store_data(“preference”, “dark_mode”)
result = client.get_data(“preference”)

# SECURITY BEST PRACTICES:

1. Token Storage:
- Never log tokens
- Store in secure client storage (encrypted)
- Transmit over HTTPS only
1. Token Lifecycle:
- Set appropriate expiration times
- Implement token refresh mechanism
- Revoke tokens on logout
1. Permissions:
- Use least privilege principle
- Validate permissions on every tool call
- Implement role-based access control
1. Session Security:
- Implement rate limiting
- Monitor for suspicious activity
- Auto-expire inactive sessions
1. Data Protection:
- Encrypt sensitive session data
- Validate all inputs
- Sanitize outputs
  “””

if **name** == “**main**”:
mcp.run()