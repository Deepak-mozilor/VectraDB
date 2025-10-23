"""Integration tests for the Python client against a running VectraDB server.

These tests require a VectraDB server to be running on localhost:50051.
Start the server with: cargo run --bin vectradb-server -- --enable-grpc
"""

import pytest
from vectradb_client import VectraDBClient, AsyncVectraDBClient


def test_server_running():
    """Verify the test server is running and accessible."""
    try:
        client = VectraDBClient()
        assert client.health_check() is True
        client.close()
    except Exception as e:
        pytest.fail(
            f"VectraDB server is not running or not accessible: {e}\n"
            "Start the server with: cargo run --bin vectradb-server -- --enable-grpc -d 3"
        )


@pytest.mark.asyncio
async def test_async_server_running():
    """Verify the test server is accessible via async client."""
    try:
        async with AsyncVectraDBClient() as client:
            assert await client.health_check() is True
    except Exception as e:
        pytest.fail(
            f"VectraDB server is not accessible via async client: {e}\n"
            "Start the server with: cargo run --bin vectradb-server -- --enable-grpc -d 3"
        )


def test_end_to_end_workflow(client, cleanup_vectors):
    """Test a complete end-to-end workflow."""
    # Create some vectors
    vec_ids = []
    for i in range(10):
        vec_id = cleanup_vectors(f"e2e_vec_{i}")
        vec_ids.append(vec_id)
        client.create_vector(
            vec_id,
            [float(i) * 0.1, float(i) * 0.2, float(i) * 0.3],
            {"index": i, "category": "test"}
        )
    
    # Verify all created
    stats = client.get_stats()
    assert stats.total_vectors >= 10
    
    # Search
    results = client.search([0.5, 1.0, 1.5], k=5)
    assert len(results) <= 5
    assert all(r.id in vec_ids for r in results)
    
    # Update one
    client.update_vector(vec_ids[0], metadata={"index": 0, "updated": True})
    vec = client.get_vector(vec_ids[0])
    assert vec.metadata["updated"] is True
    
    # List with pagination
    page1 = client.list_vectors(limit=5)
    assert len(page1) <= 5
    
    # Delete some
    for vec_id in vec_ids[:3]:
        client.delete_vector(vec_id)
    
    # Verify deletion
    final_stats = client.get_stats()
    assert final_stats.total_vectors < stats.total_vectors


@pytest.mark.asyncio
async def test_async_end_to_end_workflow(async_client, async_cleanup_vectors):
    """Test a complete end-to-end workflow with async client."""
    import asyncio
    
    # Create vectors in batch
    vec_ids = [async_cleanup_vectors(f"async_e2e_vec_{i}") for i in range(10)]
    
    await asyncio.gather(*[
        async_client.create_vector(
            vec_id,
            [float(i) * 0.1, float(i) * 0.2, float(i) * 0.3],
            {"index": i, "category": "async_test"}
        )
        for i, vec_id in enumerate(vec_ids)
    ])
    
    # Verify all created
    stats = await async_client.get_stats()
    assert stats.total_vectors >= 10
    
    # Batch search
    search_queries = [
        [0.5, 1.0, 1.5],
        [0.3, 0.6, 0.9],
        [0.7, 1.4, 2.1]
    ]
    
    results = await asyncio.gather(*[
        async_client.search(query, k=3)
        for query in search_queries
    ])
    
    assert len(results) == 3
    assert all(isinstance(r, list) for r in results)
    
    # Batch update
    await asyncio.gather(*[
        async_client.update_vector(vec_id, metadata={"batch_updated": True})
        for vec_id in vec_ids[:5]
    ])
    
    # Verify updates
    updated_vecs = await asyncio.gather(*[
        async_client.get_vector(vec_id)
        for vec_id in vec_ids[:5]
    ])
    
    assert all(v.metadata.get("batch_updated") is True for v in updated_vecs)
