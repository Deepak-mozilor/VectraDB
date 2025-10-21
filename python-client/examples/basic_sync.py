"""Basic usage example for VectraDB Python client (synchronous)."""

from vectradb_client import VectraDBClient

def main():
    # Connect to VectraDB server
    print("Connecting to VectraDB server...")
    client = VectraDBClient(host="localhost", port=50051)
    
    try:
        # Health check
        print("\n1. Performing health check...")
        if client.health_check():
            print("✓ Server is healthy!")
        
        # Create vectors
        print("\n2. Creating vectors...")
        client.create_vector("vec1", [0.1, 0.2, 0.3], {"type": "example", "index": 1})
        client.create_vector("vec2", [0.2, 0.3, 0.4], {"type": "example", "index": 2})
        client.create_vector("vec3", [0.3, 0.4, 0.5], {"type": "example", "index": 3})
        print("✓ Created 3 vectors")
        
        # Get a vector
        print("\n3. Retrieving a vector...")
        vec = client.get_vector("vec1")
        print(f"✓ Retrieved: {vec}")
        
        # Update a vector
        print("\n4. Updating vector metadata...")
        client.update_vector("vec1", metadata={"type": "updated", "index": 1, "modified": True})
        vec = client.get_vector("vec1")
        print(f"✓ Updated: {vec}")
        
        # Search for similar vectors
        print("\n5. Searching for similar vectors...")
        results = client.search(query=[0.15, 0.25, 0.35], k=3)
        print(f"✓ Found {len(results)} similar vectors:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result}")
        
        # List all vectors
        print("\n6. Listing all vectors...")
        all_vectors = client.list_vectors()
        print(f"✓ Total vectors in database: {len(all_vectors)}")
        for vec in all_vectors:
            print(f"  - {vec}")
        
        # Get database statistics
        print("\n7. Getting database statistics...")
        stats = client.get_stats()
        print(f"✓ Database stats:")
        print(f"  - Total vectors: {stats.total_vectors}")
        print(f"  - Dimension: {stats.dimension}")
        print(f"  - Index type: {stats.index_type}")
        print(f"  - Memory usage: {stats.memory_usage_mb:.2f} MB")
        
        # Upsert operation
        print("\n8. Testing upsert (update or create)...")
        client.upsert_vector("vec4", [0.4, 0.5, 0.6], {"type": "upserted"})
        print("✓ Upserted vec4")
        
        # Delete a vector
        print("\n9. Deleting a vector...")
        client.delete_vector("vec4")
        print("✓ Deleted vec4")
        
        # Final count
        stats = client.get_stats()
        print(f"\n✓ Final vector count: {stats.total_vectors}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        
    finally:
        # Clean up
        print("\nClosing connection...")
        client.close()
        print("✓ Done!")

if __name__ == "__main__":
    main()
