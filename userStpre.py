“””
FastMCP Server with User-Based Token Storage
Stores and retrieves tokens per user across tool calls
“””

from mcp.server.fastmcp import FastMCP
from typing import Optional
import json

# Initialize FastMCP server

mcp = FastMCP(“Token Storage Server”)

# In-memory storage for user tokens

# In production, use a database like Redis, PostgreSQL, etc.

user_token_store = {}

@mcp.tool()
def store_token(user_id: str, token: str, token_type: str = “access”) -> str:
“””
Store a token for a specific user

```
Args:
    user_id: Unique identifier for the user
    token: The token string to store
    token_type: Type of token (e.g., 'access', 'refresh', 'api_key')

Returns:
    Confirmation message
"""
if user_id not in user_token_store:
    user_token_store[user_id] = {}

user_token_store[user_id][token_type] = token

return f"Token stored successfully for user: {user_id}, type: {token_type}"
```

@mcp.tool()
def retrieve_token(user_id: str, token_type: str = “access”) -> str:
“””
Retrieve a stored token for a specific user

```
Args:
    user_id: Unique identifier for the user
    token_type: Type of token to retrieve

Returns:
    The stored token or error message
"""
if user_id not in user_token_store:
    return f"No tokens found for user: {user_id}"

if token_type not in user_token_store[user_id]:
    return f"No {token_type} token found for user: {user_id}"

token = user_token_store[user_id][token_type]
return f"Token retrieved: {token}"
```

@mcp.tool()
def delete_token(user_id: str, token_type: Optional[str] = None) -> str:
“””
Delete token(s) for a specific user

```
Args:
    user_id: Unique identifier for the user
    token_type: Specific token type to delete, or None to delete all tokens

Returns:
    Confirmation message
"""
if user_id not in user_token_store:
    return f"No tokens found for user: {user_id}"

if token_type:
    if token_type in user_token_store[user_id]:
        del user_token_store[user_id][token_type]
        return f"Deleted {token_type} token for user: {user_id}"
    else:
        return f"No {token_type} token found for user: {user_id}"
else:
    del user_token_store[user_id]
    return f"Deleted all tokens for user: {user_id}"
```

@mcp.tool()
def list_user_tokens(user_id: str) -> str:
“””
List all token types stored for a user

```
Args:
    user_id: Unique identifier for the user

Returns:
    JSON string of available token types
"""
if user_id not in user_token_store:
    return f"No tokens found for user: {user_id}"

token_types = list(user_token_store[user_id].keys())
return json.dumps({
    "user_id": user_id,
    "token_types": token_types,
    "count": len(token_types)
}, indent=2)
```

@mcp.tool()
def verify_and_use_token(user_id: str, api_endpoint: str, token_type: str = “access”) -> str:
“””
Example tool that retrieves and uses a stored token for an API call

```
Args:
    user_id: Unique identifier for the user
    api_endpoint: The API endpoint to call
    token_type: Type of token to use

Returns:
    Simulated API response
"""
if user_id not in user_token_store:
    return f"Error: No tokens found for user: {user_id}"

if token_type not in user_token_store[user_id]:
    return f"Error: No {token_type} token found for user: {user_id}"

token = user_token_store[user_id][token_type]

# Simulate using the token in an API call
return f"""Simulated API call:
```

Endpoint: {api_endpoint}
Authorization: Bearer {token[:10]}…
Status: 200 OK
Response: Token validated successfully for user {user_id}”””

# Alternative: Class-based approach with persistent storage

class UserTokenManager:
“””
Token manager with additional features like expiration, encryption, etc.
“””
def **init**(self):
self.storage = {}

```
def store(self, user_id: str, token: str, metadata: dict = None):
    if user_id not in self.storage:
        self.storage[user_id] = []
    
    self.storage[user_id].append({
        "token": token,
        "metadata": metadata or {},
        "created_at": "2025-12-15T00:00:00Z"  # In production, use datetime.now()
    })

def retrieve_latest(self, user_id: str):
    if user_id in self.storage and self.storage[user_id]:
        return self.storage[user_id][-1]["token"]
    return None
```

if **name** == “**main**”:
# Run the MCP server
mcp.run()