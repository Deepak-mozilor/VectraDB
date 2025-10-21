"""Async usage example for VectraDB Python client."""

import asyncio
from vectradb_client import AsyncVectraDBClient

async def main():
    # Use async context manager for automatic cleanup
    print("Connecting to VectraDB server...")
    async with AsyncVectraDBClient(host="localhost", port=50051) as client:
        
        try:
            # Health check
            print("\n1. Performing health check...")
            if await client.health_check():
                print("✓ Server is healthy!")
            
            # Create vectors
            print("\n2. Creating vectors...")
            await client.create_vector("async_vec1", [0.1, 0.2, 0.3], {"type": "async", "index": 1})
            await client.create_vector("async_vec2", [0.2, 0.3, 0.4], {"type": "async", "index": 2})
            await client.create_vector("async_vec3", [0.3, 0.4, 0.5], {"type": "async", "index": 3})
            print("✓ Created 3 vectors")
            
            # Get a vector
            print("\n3. Retrieving a vector...")
            vec = await client.get_vector("async_vec1")
            print(f"✓ Retrieved: {vec}")
            
            # Search for similar vectors
            print("\n4. Searching for similar vectors...")
            results = await client.search(query=[0.15, 0.25, 0.35], k=3)
            print(f"✓ Found {len(results)} similar vectors:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result}")
            
            # List all vectors with pagination
            print("\n5. Listing vectors with pagination...")
            page1 = await client.list_vectors(limit=2, offset=0)
            print(f"✓ Page 1 (2 vectors):")
            for vec in page1:
                print(f"  - {vec.id}: {vec.values[:3]}...")
            
            # Get database statistics
            print("\n6. Getting database statistics...")
            stats = await client.get_stats()
            print(f"✓ Database stats:")
            print(f"  - Total vectors: {stats.total_vectors}")
            print(f"  - Dimension: {stats.dimension}")
            print(f"  - Index type: {stats.index_type}")
            print(f"  - Memory usage: {stats.memory_usage_mb:.2f} MB")
            
            # Batch operations using asyncio.gather
            print("\n7. Performing batch operations...")
            await asyncio.gather(
                client.upsert_vector("batch1", [0.5, 0.6, 0.7], {"batch": True}),
                client.upsert_vector("batch2", [0.6, 0.7, 0.8], {"batch": True}),
                client.upsert_vector("batch3", [0.7, 0.8, 0.9], {"batch": True})
            )
            print("✓ Created 3 vectors in batch")
            
            # Cleanup batch vectors
            print("\n8. Cleaning up batch vectors...")
            await asyncio.gather(
                client.delete_vector("batch1"),
                client.delete_vector("batch2"),
                client.delete_vector("batch3")
            )
            print("✓ Deleted batch vectors")
            
            # Final stats
            stats = await client.get_stats()
            print(f"\n✓ Final vector count: {stats.total_vectors}")
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
    
    print("\n✓ Connection closed automatically!")

if __name__ == "__main__":
    asyncio.run(main())
