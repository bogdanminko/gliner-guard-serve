import asyncio
import time

import httpx

URL = "http://localhost:8000/predict"

TEXTS = [
    "Send $500 to John Smith at john.smith@gmail.com or I'll leak your photos",
    "My name is Alice and I live at 123 Main St, Springfield",
    "Call me at +1-555-123-4567 to discuss the contract",
    "This is a completely safe and normal message about weather",
    "Transfer money to bob@example.com immediately or else",
    "Meeting at 10am with Dr. James Wilson at 456 Oak Ave",
    "Hey, just wanted to say hello and ask about your day",
    "Send your SSN and credit card number to secure@totallylegit.com",
    "I will find where you live and make you regret this",
    "Please forward the report to maria.garcia@company.org",
    "The quick brown fox jumps over the lazy dog",
    "Wire $10000 to account 1234-5678-9012 before midnight",
    "Contact support at help@service.com for assistance",
    "You better watch your back, I know where your kids go to school",
    "Jane Doe, 789 Pine Road, Apt 4B, New York, NY 10001",
    "Let's grab coffee tomorrow at the usual place",
]

NUM_REQUESTS = 128


async def send(client: httpx.AsyncClient, text: str) -> dict:
    resp = await client.post(URL, json={"text": text})
    
    resp.raise_for_status()
    return resp.json()


async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        tasks = [send(client, TEXTS[i % len(TEXTS)]) for i in range(NUM_REQUESTS)]

        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start

    print(f"Requests : {NUM_REQUESTS}")
    print(f"Elapsed  : {elapsed:.2f}s")
    print(f"RPS      : {NUM_REQUESTS / elapsed:.1f}")
    print(f"Avg lat  : {elapsed / NUM_REQUESTS * 1000:.1f}ms")


if __name__ == "__main__":
    asyncio.run(main())
