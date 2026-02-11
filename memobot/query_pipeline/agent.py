"""
Agent class for text-based chat with memory retrieval.
"""

import requests
import json
import os
import sys
import subprocess


class Agent:
    def __init__(self, api_key, memobot_api_url="http://localhost:8000"):
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.last_retrieved_memories = []
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information.",
            }
        ]

    def _validate_openrouter_key(self) -> bool:
        """Validates the OpenRouter key by calling /auth/key."""
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=15,
            )
            if r.status_code != 200:
                print(f"OpenRouter key validation failed: HTTP {r.status_code}")
                return False
            return True
        except Exception as e:
            print(f"OpenRouter key validation error: {e}")
            return False

    def retrieve_memory(self, query):
        """Retrieve memories using the query pipeline."""
        try:
            # Direct import from the same directory
            from query import retrieve_and_rank
            
            results = retrieve_and_rank(
                question=query,
                index_name="memobot-memories",
                top_k=10,
                alpha=0.5,
                beta=0.3,
                gamma=0.2,
            )
            
            events = []
            for r in results:
                md = r.get("metadata", {})
                event = {
                    "id": r["id"],
                    "summary": md.get("summary", ""),
                    "video_file": md.get("video_file", ""),
                    "start_time_sec": md.get("start_time_sec"),
                    "end_time_sec": md.get("end_time_sec"),
                    "timestamp_utc": md.get("timestamp_utc"),
                    "importance_score": md.get("importance_score"),
                    "talking_to_camera": md.get("talking_to_camera"),
                    "relevance_score": r["relevance_score"],
                    "final_score": r["final_score"],
                }
                events.append(event)
            
            memory_context = {
                "events": events,
                "metadata": {
                    "total_results": len(events),
                    "query": query,
                }
            }
            
            self.last_retrieved_memories = memory_context.get("events", [])
            
            if self.last_retrieved_memories:
                print(f"\n📋 Retrieved {len(self.last_retrieved_memories)} relevant memories:")
                for i, event in enumerate(self.last_retrieved_memories[:3], 1):
                    print(f"  {i}. {event.get('summary', 'No summary')[:100]}...")
                if len(self.last_retrieved_memories) > 3:
                    print(f"  ... and {len(self.last_retrieved_memories) - 3} more\n")
            else:
                print("📋 No relevant memories found\n")
            
            return memory_context
            
        except ImportError as e:
            print(f"Import error: {e}")
            return {"events": [], "metadata": {"error": str(e)}}
        except Exception as e:
            print(f"Error calling retrieval pipeline: {e}")
            return {"events": [], "metadata": {"error": str(e)}}

    def get_last_memories(self):
        """Return the last retrieved memories."""
        return self.last_retrieved_memories

    def reset_conversation(self):
        """Reset the conversation history to start fresh."""
        self.last_retrieved_memories = []
        self.conversation_history = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information. When you are unsure, do not make up information.",
            }
        ]

    def chat(self, user_query):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "retrieveMemory",
                    "description": "Retrieve past memories based on a natural language query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "queryText": {
                                "type": "string",
                                "description": "The natural language query to search for memories.",
                            },
                        },
                        "required": ["queryText"],
                    },
                },
            }
        ]

        self.conversation_history.append({
            "role": "user",
            "content": user_query,
        })

        try:
            response = requests.post(
                url=self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "openai/gpt-4o",
                    "messages": self.conversation_history,
                    "tools": tools,
                    "tool_choice": "auto",
                }),
            )

            if response.status_code != 200:
                error_message = f"API returned error status {response.status_code}"
                self.conversation_history.append({"role": "assistant", "content": error_message})
                return error_message

            data = response.json()
            
            if "choices" not in data:
                error_message = "Unexpected API response format"
                self.conversation_history.append({"role": "assistant", "content": error_message})
                return error_message

            assistant_message = data["choices"][0]["message"]

            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                tool_call = assistant_message["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                if function_name == "retrieveMemory":
                    memory_context = self.retrieve_memory(function_args["queryText"])

                    self.conversation_history.append(assistant_message)
                    self.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(memory_context),
                    })

                    final_response = requests.post(
                        url=self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps({
                            "model": "openai/gpt-4o-mini",
                            "messages": self.conversation_history,
                        }),
                    )

                    if final_response.status_code != 200:
                        error_message = f"API returned error status {final_response.status_code}"
                        self.conversation_history.append({"role": "assistant", "content": error_message})
                        return error_message

                    final_data = final_response.json()
                    final_message = final_data["choices"][0]["message"]["content"]
                    self.conversation_history.append({"role": "assistant", "content": final_message})
                    return final_message

            content = assistant_message.get("content", "")
            self.conversation_history.append({"role": "assistant", "content": content})
            return content

        except Exception as error:
            print(f"Error calling OpenRouter API: {error}")
            error_message = "Sorry, I encountered an error while processing your request."
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message
