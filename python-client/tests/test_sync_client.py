"""Tests for synchronous VectraDB client."""

import pytest
from vectradb_client import VectraDBClient, VectraDBError, Vector, SearchResult, DatabaseStats


class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_success(self, client):
        """Test successful health check."""
        assert client.health_check() is True
    
    def test_health_check_bad_connection(self):
        """Test health check with bad connection."""
        bad_client = VectraDBClient(host="localhost", port=99999, timeout=1)
        with pytest.raises(VectraDBError):
            bad_client.health_check()


class TestVectorOperations:
    """Test basic vector CRUD operations."""
    
    def test_create_vector(self, client, cleanup_vectors):
        """Test creating a vector."""
        vec_id = cleanup_vectors("test_create")
        result = client.create_vector(vec_id, [0.1, 0.2, 0.3], {"test": "create"})
        assert result == vec_id
    
    def test_create_duplicate_vector(self, client, cleanup_vectors):
        """Test creating a duplicate vector raises error."""
        vec_id = cleanup_vectors("test_duplicate")
        client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        with pytest.raises(VectraDBError):
            client.create_vector(vec_id, [0.1, 0.2, 0.3])
    
    def test_get_vector(self, client, cleanup_vectors, sample_vector):
        """Test retrieving a vector."""
        vec_id = cleanup_vectors(sample_vector["id"])
        client.create_vector(vec_id, sample_vector["values"], sample_vector["metadata"])
        
        vec = client.get_vector(vec_id)
        assert isinstance(vec, Vector)
        assert vec.id == vec_id
        assert vec.values == sample_vector["values"]
        assert vec.metadata == sample_vector["metadata"]
    
    def test_get_nonexistent_vector(self, client):
        """Test getting non-existent vector raises error."""
        with pytest.raises(VectraDBError):
            client.get_vector("nonexistent")
    
    def test_update_vector(self, client, cleanup_vectors):
        """Test updating a vector."""
        vec_id = cleanup_vectors("test_update")
        client.create_vector(vec_id, [0.1, 0.2, 0.3], {"original": True})
        
        # Update metadata
        result = client.update_vector(vec_id, metadata={"updated": True})
        assert result is True
        
        # Verify update
        vec = client.get_vector(vec_id)
        assert vec.metadata.get("updated") is True
    
    def test_delete_vector(self, client, cleanup_vectors):
        """Test deleting a vector."""
        vec_id = cleanup_vectors("test_delete")
        client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        # Delete
        result = client.delete_vector(vec_id)
        assert result is True
        
        # Verify deletion
        with pytest.raises(VectraDBError):
            client.get_vector(vec_id)
    
    def test_upsert_create(self, client, cleanup_vectors):
        """Test upsert creating a new vector."""
        vec_id = cleanup_vectors("test_upsert_create")
        result = client.upsert_vector(vec_id, [0.1, 0.2, 0.3], {"upsert": "create"})
        assert result == vec_id
        
        # Verify creation
        vec = client.get_vector(vec_id)
        assert vec.id == vec_id
    
    def test_upsert_update(self, client, cleanup_vectors):
        """Test upsert updating an existing vector."""
        vec_id = cleanup_vectors("test_upsert_update")
        client.create_vector(vec_id, [0.1, 0.2, 0.3], {"original": True})
        
        # Upsert (update)
        result = client.upsert_vector(vec_id, [0.2, 0.3, 0.4], {"updated": True})
        assert result == vec_id
        
        # Verify update
        vec = client.get_vector(vec_id)
        assert vec.values == [0.2, 0.3, 0.4]


class TestSearch:
    """Test search functionality."""
    
    def test_search(self, client, cleanup_vectors):
        """Test basic search."""
        # Create test vectors
        for i in range(5):
            vec_id = cleanup_vectors(f"search_test_{i}")
            client.create_vector(vec_id, [0.1 * i, 0.2 * i, 0.3 * i])
        
        # Search
        results = client.search([0.1, 0.2, 0.3], k=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, 'id') and hasattr(r, 'distance') for r in results)
    
    def test_search_empty_database(self, client):
        """Test search in empty database returns empty results."""
        # This might not be empty, but we test it doesn't crash
        results = client.search([0.1, 0.2, 0.3], k=10)
        assert isinstance(results, list)
    
    def test_search_with_ef_search(self, client, cleanup_vectors):
        """Test search with ef_search parameter."""
        vec_id = cleanup_vectors("search_ef_test")
        client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        results = client.search([0.1, 0.2, 0.3], k=5, ef_search=50)
        assert isinstance(results, list)


class TestListVectors:
    """Test listing vectors."""
    
    def test_list_all_vectors(self, client, cleanup_vectors):
        """Test listing all vectors."""
        # Create test vectors
        for i in range(3):
            vec_id = cleanup_vectors(f"list_test_{i}")
            client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        vectors = client.list_vectors()
        assert isinstance(vectors, list)
        assert all(isinstance(v, Vector) for v in vectors)
        assert len(vectors) >= 3
    
    def test_list_vectors_with_pagination(self, client, cleanup_vectors):
        """Test listing vectors with pagination."""
        # Create test vectors
        for i in range(5):
            vec_id = cleanup_vectors(f"page_test_{i}")
            client.create_vector(vec_id, [0.1, 0.2, 0.3])
        
        # Get first page
        page1 = client.list_vectors(limit=2, offset=0)
        assert len(page1) <= 2
        
        # Get second page
        page2 = client.list_vectors(limit=2, offset=2)
        assert len(page2) <= 2


class TestDatabaseStats:
    """Test database statistics."""
    
    def test_get_stats(self, client):
        """Test getting database statistics."""
        stats = client.get_stats()
        
        assert isinstance(stats, DatabaseStats)
        assert hasattr(stats, 'total_vectors')
        assert hasattr(stats, 'dimension')
        assert hasattr(stats, 'index_type')
        assert hasattr(stats, 'memory_usage_bytes')
        assert stats.dimension > 0
        assert stats.total_vectors >= 0
    
    def test_stats_memory_usage_mb(self, client):
        """Test memory usage conversion to MB."""
        stats = client.get_stats()
        assert stats.memory_usage_mb >= 0
        assert stats.memory_usage_mb == stats.memory_usage_bytes / (1024 * 1024)


class TestContextManager:
    """Test context manager functionality."""
    
    def test_context_manager(self):
        """Test using client as context manager."""
        with VectraDBClient() as client:
            assert client.health_check() is True
        
        # Channel should be closed after context exit
        # (We can't easily test this, but it shouldn't raise an error)
