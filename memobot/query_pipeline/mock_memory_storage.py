class MockMemoryStorage:
    def search_memory(self, query_text):
        print(f'[Memory Storage] Searching for: "{query_text}"')
        
        # Mock memory database
        memories = [
            {
                "id": "1",
                "description": "You put your keys on the desk",
                "timestamp": "2023-10-10T15:42:00Z",
            },
            {
                "id": "2",
                "description": "You left your wallet in the car",
                "timestamp": "2023-10-10T14:30:00Z",
            },
            {
                "id": "3",
                "description": "You stored your passport in the safe",
                "timestamp": "2023-10-09T10:15:00Z",
            },
        ]

        print(f'[Memory Storage] Matching against {len(memories)} total memories')

        # Filter out common stop words
        stop_words = {'where', 'did', 'i', 'my', 'the', 'a', 'an', 'you', 'your', 'is', 'are', 'was', 'were', 'what', 'when', 'how'}
        keywords = [word.strip('?.,!') for word in query_text.lower().split() if word.strip('?.,!') not in stop_words]
        
        print(f'[Memory Storage] Keywords extracted (after filtering): {keywords}')
        
        relevant_memories = [
            memory for memory in memories
            if any(keyword in memory["description"].lower() for keyword in keywords)
        ]

        print(f'[Memory Storage] Found {len(relevant_memories)} relevant memories')

        return {
            "events": relevant_memories,
        }


mock_memory_storage = MockMemoryStorage()
