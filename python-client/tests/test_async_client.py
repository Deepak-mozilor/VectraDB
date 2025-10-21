"""Tests for asynchronous VectraDB client."""

import pytest
from vectradb_client import AsyncVectraDBClient, VectraDBError, Vector, SearchResult, DatabaseStats


@pytest.mark.asyncio
class TestAsyncHealthCheck:
    """Test async health check functionality."""
    
    async def test_health_check_success(self, async_client):
        """Test successful health check."""
        result = await async_client.health_check()
        assert result is True
    
    async def test_health_check_bad_connection(self):
        """Test health check with bad connection."""
        bad_client = AsyncVectraDBClient(host="localhost", port=99999, timeout=1)
        await bad_client.connect()
        with pytest.raises(VectraDBError):
            await bad_client.health_check()
        await bad_client.close()


@pytest.mark.asyncio
class TestAsyncVectorOperations:
    """Test basic async vector CRUD operations."""
    
    async def test_create_vector(self, async_client, async_cleanup_vectors):
        """Test creating a vector."""
        vec_id = async_cleanup_vectors("async_test_create")
        result = await async_client.create_vector(vec_id, [0.1, 0.2, 0.3], {"test": "create"})
        assert result == vec_id
    
    async def test_create_duplicate_vector(self, async_client, async_cleanup_vectors):
        """Test creating a duplicate vector raises error."""
        vec_id = async_cleanup_vectors("async_test_duplicate")
        await async_client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        with pytest.raises(VectraDBError):
            await async_client.create_vector(vec_id, [0.1, 0.2, 0.3])
    
    async def test_get_vector(self, async_client, async_cleanup_vectors, sample_vector):
        """Test retrieving a vector."""
        vec_id = async_cleanup_vectors(f"async_{sample_vector['id']}")
        await async_client.create_vector(vec_id, sample_vector["values"], sample_vector["metadata"])
        
        vec = await async_client.get_vector(vec_id)
        assert isinstance(vec, Vector)
        assert vec.id == vec_id
        assert vec.values == sample_vector["values"]
        assert vec.metadata == sample_vector["metadata"]
    
    async def test_get_nonexistent_vector(self, async_client):
        """Test getting non-existent vector raises error."""
        with pytest.raises(VectraDBError):
            await async_client.get_vector("async_nonexistent")
    
    async def test_update_vector(self, async_client, async_cleanup_vectors):
        """Test updating a vector."""
        vec_id = async_cleanup_vectors("async_test_update")
        await async_client.create_vector(vec_id, [0.1, 0.2, 0.3], {"original": True})
        
        # Update metadata
        result = await async_client.update_vector(vec_id, metadata={"updated": True})
        assert result is True
        
        # Verify update
        vec = await async_client.get_vector(vec_id)
        assert vec.metadata.get("updated") is True
    
    async def test_delete_vector(self, async_client, async_cleanup_vectors):
        """Test deleting a vector."""
        vec_id = async_cleanup_vectors("async_test_delete")
        await async_client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        # Delete
        result = await async_client.delete_vector(vec_id)
        assert result is True
        
        # Verify deletion
        with pytest.raises(VectraDBError):
            await async_client.get_vector(vec_id)
    
    async def test_upsert_create(self, async_client, async_cleanup_vectors):
        """Test upsert creating a new vector."""
        vec_id = async_cleanup_vectors("async_test_upsert_create")
        result = await async_client.upsert_vector(vec_id, [0.1, 0.2, 0.3], {"upsert": "create"})
        assert result == vec_id
        
        # Verify creation
        vec = await async_client.get_vector(vec_id)
        assert vec.id == vec_id
    
    async def test_upsert_update(self, async_client, async_cleanup_vectors):
        """Test upsert updating an existing vector."""
        vec_id = async_cleanup_vectors("async_test_upsert_update")
        await async_client.create_vector(vec_id, [0.1, 0.2, 0.3], {"original": True})
        
        # Upsert (update)
        result = await async_client.upsert_vector(vec_id, [0.2, 0.3, 0.4], {"updated": True})
        assert result == vec_id
        
        # Verify update
        vec = await async_client.get_vector(vec_id)
        assert vec.values == [0.2, 0.3, 0.4]


@pytest.mark.asyncio
class TestAsyncSearch:
    """Test async search functionality."""
    
    async def test_search(self, async_client, async_cleanup_vectors):
        """Test basic search."""
        # Create test vectors
        for i in range(5):
            vec_id = async_cleanup_vectors(f"async_search_test_{i}")
            await async_client.create_vector(vec_id, [0.1 * i, 0.2 * i, 0.3 * i])
        
        # Search
        results = await async_client.search([0.1, 0.2, 0.3], k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, 'id') and hasattr(r, 'distance') for r in results)
    
    async def test_search_empty_database(self, async_client):
        """Test search in empty database returns empty results."""
        results = await async_client.search([0.1, 0.2, 0.3], k=10)
        assert isinstance(results, list)
    
    async def test_search_with_ef_search(self, async_client, async_cleanup_vectors):
        """Test search with ef_search parameter."""
        vec_id = async_cleanup_vectors("async_search_ef_test")
        await async_client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        results = await async_client.search([0.1, 0.2, 0.3], k=5, ef_search=50)
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestAsyncListVectors:
    """Test async listing vectors."""
    
    async def test_list_all_vectors(self, async_client, async_cleanup_vectors):
        """Test listing all vectors."""
        # Create test vectors
        for i in range(3):
            vec_id = async_cleanup_vectors(f"async_list_test_{i}")
            await async_client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        vectors = await async_client.list_vectors()
        assert isinstance(vectors, list)
        assert all(isinstance(v, Vector) for v in vectors)
        assert len(vectors) >= 3
    
    async def test_list_vectors_with_pagination(self, async_client, async_cleanup_vectors):
        """Test listing vectors with pagination."""
        # Create test vectors
        for i in range(5):
            vec_id = async_cleanup_vectors(f"async_page_test_{i}")
            await async_client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        # Get first page
        page1 = await async_client.list_vectors(limit=2, offset=0)
        assert len(page1) <= 2
        
        # Get second page
        page2 = await async_client.list_vectors(limit=2, offset=2)
        assert len(page2) <= 2


@pytest.mark.asyncio
class TestAsyncDatabaseStats:
    """Test async database statistics."""
    
    async def test_get_stats(self, async_client):
        """Test getting database statistics."""
        stats = await async_client.get_stats()
        
        assert isinstance(stats, DatabaseStats)
        assert hasattr(stats, 'total_vectors')
        assert hasattr(stats, 'dimension')
        assert hasattr(stats, 'index_type')
        assert hasattr(stats, 'memory_usage_bytes')
        assert stats.dimension > 0
        assert stats.total_vectors >= 0
    
    async def test_stats_memory_usage_mb(self, async_client):
        """Test memory usage conversion to MB."""
        stats = await async_client.get_stats()
        assert stats.memory_usage_mb >= 0
        assert stats.memory_usage_mb == stats.memory_usage_bytes / (1024 * 1024)


@pytest.mark.asyncio
class TestAsyncContextManager:
    """Test async context manager functionality."""
    
    async def test_context_manager(self):
        """Test using async client as context manager."""
        async with AsyncVectraDBClient() as client:
            result = await client.health_check()
            assert result is True
        
        # Channel should be closed after context exit


@pytest.mark.asyncio
class TestAsyncBatchOperations:
    """Test batch operations using asyncio.gather."""
    
    async def test_batch_create(self, async_client, async_cleanup_vectors):
        """Test creating multiple vectors in batch."""
        import asyncio
        
        ids = [async_cleanup_vectors(f"batch_{i}") for i in range(3)]
        
        # Create in batch
        await asyncio.gather(*[
            async_client.create_vector(vec_id, [0.1 * i, 0.2 * i, 0.3 * i])
            for i, vec_id in enumerate(ids)
        ])
        
        # Verify all created
        for vec_id in ids:
            vec = await async_client.get_vector(vec_id)
            assert vec.id == vec_id
