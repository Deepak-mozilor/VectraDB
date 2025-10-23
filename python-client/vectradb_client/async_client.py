"""Asynchronous VectraDB client implementation."""

import json
from typing import List, Dict, Any, Optional
import grpc.aio

from .types import Vector, SearchResult, DatabaseStats, VectraDBError

# Import generated gRPC code
try:
    from . import vectradb_pb2
    from . import vectradb_pb2_grpc
except ImportError:
    raise ImportError(
        "gRPC stubs not found. Please run 'python generate_proto.py' first to generate them."
    )


class AsyncVectraDBClient:
    """Asynchronous client for VectraDB vector database.
    
    This client uses async gRPC to communicate with the VectraDB Rust backend server.
    All methods are async and should be awaited.
    
    Args:
        host: Server host (default: "localhost")
        port: Server gRPC port (default: 50051)
        timeout: Request timeout in seconds (default: 30)
        
    Example:
        >>> async with AsyncVectraDBClient() as client:
        ...     await client.create_vector("vec1", [0.1, 0.2, 0.3])
        ...     results = await client.search([0.1, 0.2, 0.3], k=10)
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051, timeout: int = 30):
        """Initialize the async VectraDB client."""
        self.host = host
        self.port = port
        self.timeout = timeout
        self.address = f"{host}:{port}"
        self.channel = None
        self.stub = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def connect(self):
        """Establish connection to the server."""
        self.channel = grpc.aio.insecure_channel(self.address)
        self.stub = vectradb_pb2_grpc.VectraDbStub(self.channel)
        
    async def close(self):
        """Close the gRPC channel."""
        if self.channel:
            await self.channel.close()
            
    def _metadata_to_dict(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata dict to JSON string."""
        return json.dumps(metadata) if metadata else "{}"
        
    def _dict_to_metadata(self, metadata_str: str) -> Dict[str, Any]:
        """Convert JSON string to metadata dict."""
        try:
            return json.loads(metadata_str) if metadata_str else {}
        except json.JSONDecodeError:
            return {}
            
    async def create_vector(
        self, 
        id: str, 
        values: List[float], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new vector in the database.
        
        Args:
            id: Unique identifier for the vector
            values: Vector values (must match database dimension)
            metadata: Optional metadata dictionary
            
        Returns:
            The created vector ID
            
        Raises:
            VectraDBError: If the vector already exists or other error occurs
        """
        try:
            request = vectradb_pb2.CreateVectorRequest(
                id=id,
                values=values,
                metadata=self._metadata_to_dict(metadata or {})
            )
            response = await self.stub.CreateVector(request, timeout=self.timeout)
            
            if not response.success:
                raise VectraDBError(response.message)
                
            return response.id
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def get_vector(self, id: str) -> Vector:
        """Get a vector by ID.
        
        Args:
            id: Vector ID to retrieve
            
        Returns:
            Vector object with id, values, and metadata
            
        Raises:
            VectraDBError: If vector not found or other error occurs
        """
        try:
            request = vectradb_pb2.GetVectorRequest(id=id)
            response = await self.stub.GetVector(request, timeout=self.timeout)
            
            if not response.found:
                raise VectraDBError(f"Vector with id '{id}' not found")
                
            return Vector(
                id=response.vector.id,
                values=list(response.vector.values),
                metadata=self._dict_to_metadata(response.vector.metadata)
            )
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def update_vector(
        self,
        id: str,
        values: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update an existing vector's values and/or metadata.
        
        Args:
            id: Vector ID to update
            values: New vector values (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if update was successful
            
        Raises:
            VectraDBError: If vector not found or update fails
        """
        try:
            request = vectradb_pb2.UpdateVectorRequest(
                id=id,
                values=values or [],
                metadata=self._metadata_to_dict(metadata or {})
            )
            response = await self.stub.UpdateVector(request, timeout=self.timeout)
            
            if not response.success:
                raise VectraDBError(response.message)
                
            return response.success
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def delete_vector(self, id: str) -> bool:
        """Delete a vector by ID.
        
        Args:
            id: Vector ID to delete
            
        Returns:
            True if deletion was successful
            
        Raises:
            VectraDBError: If vector not found or deletion fails
        """
        try:
            request = vectradb_pb2.DeleteVectorRequest(id=id)
            response = await self.stub.DeleteVector(request, timeout=self.timeout)
            
            if not response.success:
                raise VectraDBError(response.message)
                
            return response.success
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def upsert_vector(
        self,
        id: str,
        values: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create or update a vector (upsert operation).
        
        Args:
            id: Vector ID
            values: Vector values
            metadata: Optional metadata
            
        Returns:
            The vector ID
            
        Raises:
            VectraDBError: If operation fails
        """
        try:
            request = vectradb_pb2.UpsertVectorRequest(
                id=id,
                values=values,
                metadata=self._metadata_to_dict(metadata or {})
            )
            response = await self.stub.UpsertVector(request, timeout=self.timeout)
            
            if not response.success:
                raise VectraDBError(response.message)
                
            return response.id
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def search(
        self,
        query: List[float],
        k: int = 10,
        ef_search: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for k nearest neighbors of the query vector.
        
        Args:
            query: Query vector
            k: Number of results to return (default: 10)
            ef_search: HNSW search parameter (optional)
            
        Returns:
            List of SearchResult objects sorted by distance
            
        Raises:
            VectraDBError: If search fails
        """
        try:
            request = vectradb_pb2.SearchRequest(
                query=query,
                k=k,
                ef_search=ef_search if ef_search is not None else 0
            )
            response = await self.stub.Search(request, timeout=self.timeout)
            
            results = []
            for result in response.results:
                results.append(SearchResult(
                    id=result.id,
                    distance=result.distance
                ))
                
            return results
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def list_vectors(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[Vector]:
        """List all vectors in the database with pagination.
        
        Args:
            limit: Maximum number of vectors to return (optional)
            offset: Number of vectors to skip (optional)
            
        Returns:
            List of Vector objects
            
        Raises:
            VectraDBError: If listing fails
        """
        try:
            request = vectradb_pb2.ListVectorsRequest(
                limit=limit if limit is not None else 0,
                offset=offset if offset is not None else 0
            )
            response = await self.stub.ListVectors(request, timeout=self.timeout)
            
            vectors = []
            for vec in response.vectors:
                vectors.append(Vector(
                    id=vec.id,
                    values=list(vec.values),
                    metadata=self._dict_to_metadata(vec.metadata)
                ))
                
            return vectors
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def get_stats(self) -> DatabaseStats:
        """Get database statistics.
        
        Returns:
            DatabaseStats object with database information
            
        Raises:
            VectraDBError: If request fails
        """
        try:
            request = vectradb_pb2.GetStatsRequest()
            response = await self.stub.GetStats(request, timeout=self.timeout)
            
            return DatabaseStats(
                total_vectors=response.total_vectors,
                dimension=response.dimension,
                index_type=response.index_type,
                memory_usage_bytes=response.memory_usage_bytes
            )
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
            
    async def health_check(self) -> bool:
        """Check if the server is healthy and responding.
        
        Returns:
            True if server is healthy
            
        Raises:
            VectraDBError: If health check fails
        """
        try:
            request = vectradb_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request, timeout=self.timeout)
            return response.healthy
            
        except grpc.RpcError as e:
            raise VectraDBError(f"gRPC error: {e.details()}") from e
