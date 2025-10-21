"""Example showing context manager usage for both sync and async clients."""

from vectradb_client import VectraDBClient, AsyncVectraDBClient
import asyncio

def sync_example():
    """Synchronous context manager example."""
    print("=== Synchronous Context Manager Example ===\n")
    
    # Context manager automatically closes connection
    with VectraDBClient(host="localhost", port=50051) as client:
        # Create vector
        client.create_vector("ctx_sync", [1.0, 2.0, 3.0], {"method": "sync"})
        
        # Get stats
        stats = client.get_stats()
        print(f"Total vectors: {stats.total_vectors}")
        print(f"Dimension: {stats.dimension}")
        
        # Search
        results = client.search([1.0, 2.0, 3.0], k=5)
        print(f"Found {len(results)} similar vectors")
        
        # Cleanup
        client.delete_vector("ctx_sync")
    
    print("✓ Sync context manager closed automatically\n")

async def async_example():
    """Asynchronous context manager example."""
    print("=== Asynchronous Context Manager Example ===\n")
    
    # Async context manager automatically closes connection
    async with AsyncVectraDBClient(host="localhost", port=50051) as client:
        # Create vector
        await client.create_vector("ctx_async", [1.0, 2.0, 3.0], {"method": "async"})
        
        # Get stats
        stats = await client.get_stats()
        print(f"Total vectors: {stats.total_vectors}")
        print(f"Dimension: {stats.dimension}")
        
        # Search
        results = await client.search([1.0, 2.0, 3.0], k=5)
        print(f"Found {len(results)} similar vectors")
        
        # Cleanup
        await client.delete_vector("ctx_async")
    
    print("✓ Async context manager closed automatically\n")

def main():
    """Run both examples."""
    # Sync example
    sync_example()
    
    # Async example
    asyncio.run(async_example())
    
    print("✓ All examples completed!")

if __name__ == "__main__":
    main()
