from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

transport = RequestsHTTPTransport(
    url="https://api.example.com/graphql",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    verify=True
)

client = Client(transport=transport, fetch_schema_from_transport=True)

query = gql("""
  query {
    users {
      id
      name
    }
  }
""")

result = client.execute(query)
print(result)

response = requests.post(url, json={"query": query}, headers=headers)
response.raise_for_status()

payload = response.json()

if "errors" in payload:
    for err in payload["errors"]:
        print(f"GraphQL error: {err['message']}")
else:
    data = payload["data"]

