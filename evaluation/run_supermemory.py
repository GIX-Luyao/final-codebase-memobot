# # pip install supermemory
# from supermemory import Supermemory

# client = Supermemory(
#     api_key="sm_3BYwUBUd7BZgjs5Ls4WnKc_MTesHLacWmrQCacHmYgVoSQWmjBqlLbmdusNuKGjORcrlAtyAmPArePQalRNcoqc",
# )

# response = client.memories.add(
#     content="SuperMemory Python SDK is awesome.",
#     container_tag="Python_SDK",
#     metadata={
#         "note_id": "123",
#     }
# )
# print(response)

# searching = client.search.execute(
#     q="What do you know about me?",
# )
# print(searching.results)

# from supermemory import Supermemory

# client = Supermemory()
# USER_ID = "dhravya"

# conversation = [
#     {"role": "assistant", "content": "Hello, how are you doing?"},
#     {"role": "user", "content": "Hello! I am Dhravya. I am 20 years old. I love to code!"},
#     {"role": "user", "content": "Can I go to the club?"},
# ]

# # Get user profile + relevant memories for context
# profile = client.profile(container_tag=USER_ID, q=conversation[-1]["content"])

# static = "\n".join(profile.profile.static)
# dynamic = "\n".join(profile.profile.dynamic)
# memories = "\n".join(r.get("memory", "") for r in profile.search_results.results)

# context = f"""Static profile:
# {static}

# Dynamic profile:
# {dynamic}

# Relevant memories:
# {memories}"""

# # Build messages with memory-enriched context
# messages = [{"role": "system", "content": f"User context:\n{context}"}, *conversation]

# # response = llm.chat(messages=messages)

# # Store conversation for future context
# client.add(
#     content="\n".join(f"{m['role']}: {m['content']}" for m in conversation),
#     container_tag=USER_ID,
# )

import os
from supermemory import Supermemory

client = Supermemory(
    api_key="sm_3BYwUBUd7BZgjs5Ls4WnKc_MTesHLacWmrQCacHmYgVoSQWmjBqlLbmdusNuKGjORcrlAtyAmPArePQalRNcoqc"
)

# Add a memory
client.add(content="Meeting notes from Q1 planning", container_tags=["user_123"])

# Search memories
response = client.search.documents(
    q="planning notes",
    container_tags=["user_123"]
)
print(response.results)

# Get user profile
profile = client.profile(container_tag="user_123")
print(profile.profile.static)
print(profile.profile.dynamic)

# Add with metadata
client.add(
    content="Technical design doc",
    container_tags=["user_123"],
    metadata={"category": "engineering", "priority": "high"}
)

# Search with filters
results = client.search.documents(
    q="design document",
    container_tags=["user_123"],
    filters={
        "AND": [
            {"key": "category", "value": "engineering"}
        ]
    }
)
print(results.results)

# List documents
docs = client.documents.list(container_tags=["user_123"], limit=10)

print(docs)