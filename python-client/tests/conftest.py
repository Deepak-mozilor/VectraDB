"""Test configuration and fixtures."""

import pytest
from vectradb_client import VectraDBClient, AsyncVectraDBClient

# Test server configuration
TEST_HOST = "localhost"
TEST_PORT = 50051
TEST_TIMEOUT = 10


@pytest.fixture
def client():
    """Create a synchronous test client."""
    client = VectraDBClient(host=TEST_HOST, port=TEST_PORT, timeout=TEST_TIMEOUT)
    yield client
    client.close()


@pytest.fixture
async def async_client():
    """Create an asynchronous test client."""
    client = AsyncVectraDBClient(host=TEST_HOST, port=TEST_PORT, timeout=TEST_TIMEOUT)
    await client.connect()
    yield client
    await client.close()


@pytest.fixture
def sample_vector():
    """Sample vector data for tests."""
    return {
        "id": "test_vec_1",
        "values": [0.1, 0.2, 0.3],
        "metadata": {"type": "test", "index": 1}
    }


@pytest.fixture
def cleanup_vectors(client):
    """Cleanup test vectors after test."""
    created_ids = []
    
    def _track_id(vec_id):
        created_ids.append(vec_id)
        return vec_id
    
    yield _track_id
    
    # Cleanup
    for vec_id in created_ids:
        try:
            client.delete_vector(vec_id)
        except Exception:
            pass  # Ignore errors during cleanup


@pytest.fixture
async def async_cleanup_vectors(async_client):
    """Cleanup test vectors after async test."""
    created_ids = []
    
    def _track_id(vec_id):
        created_ids.append(vec_id)
        return vec_id
    
    yield _track_id
    
    # Cleanup
    for vec_id in created_ids:
        try:
            await async_client.delete_vector(vec_id)
        except Exception:
            pass  # Ignore errors during cleanup
