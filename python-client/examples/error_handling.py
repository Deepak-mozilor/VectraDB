"""Example showing error handling patterns."""

from vectradb_client import VectraDBClient, VectraDBError

def main():
    """Demonstrate error handling patterns."""
    print("=== Error Handling Examples ===\n")
    
    client = VectraDBClient(host="localhost", port=50051)
    
    try:
        # 1. Vector not found
        print("1. Testing vector not found error...")
        try:
            client.get_vector("nonexistent_vector")
        except VectraDBError as e:
            print(f"✓ Caught expected error: {e}\n")
        
        # 2. Duplicate vector creation
        print("2. Testing duplicate vector error...")
        client.create_vector("duplicate_test", [1.0, 2.0, 3.0])
        try:
            client.create_vector("duplicate_test", [1.0, 2.0, 3.0])
        except VectraDBError as e:
            print(f"✓ Caught expected error: {e}\n")
        finally:
            client.delete_vector("duplicate_test")
        
        # 3. Connection error (using wrong port)
        print("3. Testing connection error...")
        try:
            bad_client = VectraDBClient(host="localhost", port=99999)
            bad_client.health_check()
        except VectraDBError as e:
            print(f"✓ Caught connection error: {e}\n")
        
        # 4. Dimension mismatch
        print("4. Testing dimension mismatch...")
        stats = client.get_stats()
        print(f"Server dimension: {stats.dimension}")
        
        wrong_dim = [1.0, 2.0]  # Wrong dimension
        try:
            client.create_vector("wrong_dim", wrong_dim)
        except VectraDBError as e:
            print(f"✓ Caught dimension error: {e}\n")
        
        # 5. Graceful handling with fallbacks
        print("5. Testing graceful fallback pattern...")
        try:
            vec = client.get_vector("maybe_exists")
            print(f"Vector found: {vec.id}")
        except VectraDBError:
            print("Vector not found, creating new one...")
            client.upsert_vector("maybe_exists", [1.0, 2.0, 3.0])
            print("✓ Created new vector\n")
        finally:
            client.delete_vector("maybe_exists")
        
        # 6. Retry pattern
        print("6. Testing retry pattern...")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.health_check()
                print(f"✓ Health check succeeded on attempt {attempt + 1}\n")
                break
            except VectraDBError as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed, retrying...")
                else:
                    print(f"✗ All attempts failed: {e}\n")
                    raise
        
        print("✓ All error handling tests completed!")
        
    finally:
        client.close()

if __name__ == "__main__":
    main()
