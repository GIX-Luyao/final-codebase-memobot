import requests
import json
import os
from mock_memory_storage import mock_memory_storage


class Agent:
    def __init__(self, memory_storage, api_key):
        self.memory_storage = memory_storage
        self.api_key = api_key
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def retrieve_memory(self, query):
        print(f'[Memory API] Retrieving memory for query: "{query}"')
        memory_context = self.memory_storage.search_memory(query_text=query)
        print(f'[Memory API] Found {len(memory_context["events"])} relevant memory(ies)')
        return memory_context

    def chat(self, user_query):
        print(f'[Agent] Received user query: "{user_query}"')
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
                                "description": "The natural language query to search for memories",
                            },
                        },
                        "required": ["queryText"],
                    },
                },
            }
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant with access to a memory system. When users ask about past events, use the retrieveMemory function to find relevant information.",
            },
            {
                "role": "user",
                "content": user_query,
            },
        ]

        try:
            print(f'[Agent] Calling GPT API with tools...')
            # First API call to get tool usage
            response = requests.post(
                url=self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": "openai/gpt-4o-mini",
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                }),
            )

            data = response.json()
            assistant_message = data["choices"][0]["message"]

            # Check if the model wants to call a function
            if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
                tool_call = assistant_message["tool_calls"][0]
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])

                print(f'[Agent] GPT wants to call function: {function_name}')
                print(f'[Agent] Function arguments: {function_args}')

                if function_name == "retrieveMemory":
                    # Execute the memory retrieval
                    memory_context = self.retrieve_memory(function_args["queryText"])

                    print(f'[Agent] Memory context retrieved:')
                    print(json.dumps(memory_context, indent=2))

                    # Add the function result to messages
                    messages.append(assistant_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(memory_context),
                    })

                    print(f'[Agent] Calling GPT API with memory context...')
                    # Second API call with function result
                    final_response = requests.post(
                        url=self.api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        data=json.dumps({
                            "model": "openai/gpt-4o-mini",
                            "messages": messages,
                        }),
                    )

                    final_data = final_response.json()
                    print(f'[Agent] Final response generated')
                    return final_data["choices"][0]["message"]["content"]

            return assistant_message.get("content", "")

        except Exception as error:
            print(f"[Agent] Error calling OpenRouter API: {error}")
            return "Sorry, I encountered an error while processing your request."


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Starting Memobot Agent Test")
    print("=" * 60)
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    agent = Agent(mock_memory_storage, api_key)

    user_query = "Where did I put my keys?"
    print(f"\n[User] {user_query}\n")

    response = agent.chat(user_query)
    print(f"\n[Agent Response] {response}\n")
    
    print("=" * 60)
