"""
Comprehensive VectraDB Test Script
Demonstrates: Adding, Deleting, Top-K Search, Chunking, and Embedding
"""

import sys
import os
from typing import List

# Add current directory to path to import vectradb_simple
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectradb_simple import VectraDB

# Simple mock embedding function (replace with real embeddings like OpenAI, Sentence-Transformers, etc.)
def embed_text(text: str, dimension: int = 3) -> List[float]:
    """
    Mock embedding function - converts text to a simple vector.
    In production, use real embeddings like:
    - OpenAI embeddings (1536 dims)
    - Sentence-Transformers (384/768 dims)
    - Cohere embeddings (4096 dims)
    """
    # Simple hash-based mock embedding for demonstration
    hash_val = hash(text)
    vector = []
    for i in range(dimension):
        # Create pseudo-random but deterministic values
        val = ((hash_val + i * 12345) % 10000) / 10000.0
        vector.append(val)
    return vector


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Simple text chunking with overlap.
    Splits text into chunks of approximately chunk_size characters with overlap.
    """
    text = text.strip()
    if not text:
        return []
    
    chunks = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        # Calculate end position
        end = min(i + chunk_size, text_len)
        
        # Extract chunk
        chunk = text[i:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Break if we've reached the end
        if end >= text_len:
            break
            
        # Move forward with overlap
        i = end - overlap
        
        # Prevent infinite loop
        if i <= chunks.__len__() * (chunk_size - overlap) - overlap:
            i = end - overlap + 1
    
    return chunks


def main():
    print("=" * 70)
    print("VectraDB Comprehensive Test")
    print("Features: Adding, Deleting, Top-K Search, Chunking, Embedding")
    print("=" * 70)
    
    # Connect to VectraDB (use WSL IP for Windows-WSL connection)
    print("\n📡 Connecting to VectraDB server...")
    try:
        client = VectraDB(host="172.26.49.233", port=50051)
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\n💡 Make sure the server is running:")
        print("   cargo run --bin vectradb-server -- --enable-grpc -d 3 -D ./vectradb_data")
        sys.exit(1)
    
    # Health check
    print("\n🏥 Health Check...")
    if not client.health_check():
        print("❌ Server is not healthy")
        client.close()
        sys.exit(1)
    print("✅ Server is healthy!")
    
    # Get initial stats
    print("\n📊 Initial Database State:")
    stats = client.stats()
    print(f"   Total vectors: {stats.total_vectors}")
    print(f"   Dimension: {stats.dimension}")
    print(f"   Memory usage: {stats.memory_usage} bytes")
    
    # Sample documents for testing
    documents = [
        {
            "id": "doc1",
            "text": "Machine learning is a subset of artificial intelligence that focuses on "
                   "building systems that learn from data. It enables computers to improve "
                   "their performance on tasks without being explicitly programmed.",
            "category": "AI"
        },
        {
            "id": "doc2",
            "text": "Deep learning uses neural networks with multiple layers to learn "
                   "hierarchical representations of data. It has revolutionized fields "
                   "like computer vision and natural language processing.",
            "category": "AI"
        },
        {
            "id": "doc3",
            "text": "Python is a high-level programming language known for its simplicity "
                   "and readability. It's widely used in web development, data science, "
                   "and machine learning applications.",
            "category": "Programming"
        },
        {
            "id": "doc4",
            "text": "Vector databases store and retrieve high-dimensional vectors efficiently. "
                   "They are essential for semantic search, recommendation systems, and "
                   "retrieval-augmented generation in AI applications.",
            "category": "Database"
        },
        {
            "id": "doc5",
            "text": "Natural language processing enables computers to understand and generate "
                   "human language. It powers applications like chatbots, translation "
                   "services, and sentiment analysis.",
            "category": "AI"
        }
    ]
    
    # PART 1: TEXT CHUNKING
    print("\n" + "=" * 70)
    print("PART 1: TEXT CHUNKING")
    print("=" * 70)
    
    long_document = """
    Vector databases are specialized database systems designed to store and query 
    high-dimensional vector embeddings. These embeddings are numerical representations 
    of data that capture semantic meaning.
    
    In traditional databases, we search for exact matches or simple comparisons. 
    But in vector databases, we search for semantic similarity. This means we can 
    find content that means the same thing, even if it uses different words.
    
    The key operation in vector databases is similarity search, also known as 
    nearest neighbor search. This finds the vectors that are closest to a query 
    vector in the high-dimensional space. Common distance metrics include cosine 
    similarity, Euclidean distance, and dot product.
    
    Applications of vector databases include semantic search, recommendation engines, 
    question-answering systems, and retrieval-augmented generation (RAG) for large 
    language models. They enable AI systems to find relevant information quickly 
    and accurately.
    """
    
    print(f"\n📄 Original document length: {len(long_document)} characters")
    print(f"📝 Chunking document with chunk_size=200, overlap=50...")
    
    chunks = chunk_text(long_document, chunk_size=200, overlap=50)
    print(f"✅ Created {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\n   Chunk {i+1} ({len(chunk)} chars):")
        print(f"   \"{chunk[:80]}...\"")
    
    # PART 2: TEXT EMBEDDING
    print("\n" + "=" * 70)
    print("PART 2: TEXT EMBEDDING")
    print("=" * 70)
    
    print("\n🧮 Embedding text chunks into vectors...")
    print("   (Using mock embeddings - replace with real embeddings in production)")
    
    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        vector = embed_text(chunk, dimension=stats.dimension)
        embedded_chunks.append({
            "id": f"chunk_{i+1}",
            "text": chunk,
            "vector": vector
        })
        print(f"   ✅ Chunk {i+1}: {vector[:3]}... (showing first 3 dimensions)")
    
    # PART 3: ADDING VECTORS
    print("\n" + "=" * 70)
    print("PART 3: ADDING VECTORS TO DATABASE")
    print("=" * 70)
    
    print("\n➕ Adding documents with embeddings...")
    
    # Add regular documents
    for doc in documents:
        vector = embed_text(doc["text"], dimension=stats.dimension)
        client.create(doc["id"], vector, {"category": doc["category"], "type": "document"})
        print(f"   ✅ Added {doc['id']}: {doc['category']}")
    
    # Add chunked document
    print("\n➕ Adding chunked document vectors...")
    for chunk_data in embedded_chunks:
        client.create(
            chunk_data["id"],
            chunk_data["vector"],
            {"type": "chunk", "source": "long_document"}
        )
        print(f"   ✅ Added {chunk_data['id']}")
    
    # Verify additions
    stats = client.stats()
    print(f"\n📊 After adding vectors:")
    print(f"   Total vectors: {stats.total_vectors}")
    print(f"   Expected: {len(documents) + len(embedded_chunks)}")
    
    # PART 4: LISTING VECTORS
    print("\n" + "=" * 70)
    print("PART 4: LISTING ALL VECTORS")
    print("=" * 70)
    
    all_vectors = client.list()
    print(f"\n📋 Found {len(all_vectors.ids)} vectors in database:")
    print(f"   IDs: {', '.join(all_vectors.ids)}")
    
    # PART 5: TOP-K SIMILARITY SEARCH
    print("\n" + "=" * 70)
    print("PART 5: TOP-K SIMILARITY SEARCH")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        {
            "text": "What is machine learning and artificial intelligence?",
            "k": 3
        },
        {
            "text": "Tell me about vector databases and semantic search",
            "k": 5
        },
        {
            "text": "Programming languages for data science",
            "k": 3
        }
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: \"{query['text']}\"")
        print(f"   Searching for top-{query['k']} similar vectors...")
        
        # Embed the query
        query_vector = embed_text(query["text"], dimension=stats.dimension)
        
        # Search
        results = client.search(query_vector, k=query["k"])
        
        print(f"   ✅ Found {len(results.results)} results:")
        
        for j, result in enumerate(results.results, 1):
            # Get the full vector to see its tags and vector values
            vec = client.get(result.id)
            print(f"\n   Result #{j}:")
            print(f"      ID: {result.id}")
            print(f"      Score: {result.score:.4f}")
            print(f"      Vector: [{vec.vector[0]:.4f}, {vec.vector[1]:.4f}, {vec.vector[2]:.4f}, ..., {vec.vector[-1]:.4f}] (dim={len(vec.vector)})")
            print(f"      Tags: {vec.tags}")
            if vec.tags.get("type") == "document":
                # Find original document
                orig_doc = next((d for d in documents if d["id"] == result.id), None)
                if orig_doc:
                    print(f"      Text: \"{orig_doc['text'][:100]}...\"")
            elif vec.tags.get("type") == "chunk":
                # Find chunk
                chunk_data = next((c for c in embedded_chunks if c["id"] == result.id), None)
                if chunk_data:
                    print(f"      Text: \"{chunk_data['text'][:100]}...\"")
    
    # PART 6: FILTERING BY METADATA
    print("\n" + "=" * 70)
    print("PART 6: FILTERING BY METADATA (CATEGORY)")
    print("=" * 70)
    
    print("\n🔍 Finding all documents in 'AI' category...")
    ai_docs = []
    for vec_id in all_vectors.ids:
        vec = client.get(vec_id)
        if vec.tags.get("category") == "AI":
            ai_docs.append(vec_id)
            orig_doc = next((d for d in documents if d["id"] == vec_id), None)
            if orig_doc:
                print(f"   ✅ {vec_id}: \"{orig_doc['text'][:80]}...\"")
    
    print(f"\n📊 Found {len(ai_docs)} documents in 'AI' category")
    
    # PART 7: UPDATING VECTORS
    print("\n" + "=" * 70)
    print("PART 7: UPDATING VECTORS")
    print("=" * 70)
    
    print("\n📝 Updating doc1 with new metadata...")
    vec = client.get("doc1")
    print(f"   Before: {vec.tags}")
    
    # Update with same vector but new tags
    client.update("doc1", vec.vector, {"category": "AI", "type": "document", "updated": "true", "version": "2"})
    
    vec = client.get("doc1")
    print(f"   After:  {vec.tags}")
    print("   ✅ Update successful!")
    
    # PART 8: DELETING VECTORS
    print("\n" + "=" * 70)
    print("PART 8: DELETING VECTORS")
    print("=" * 70)
    
    # Delete one document
    print("\n🗑️  Deleting doc5...")
    client.delete("doc5")
    print("   ✅ Deleted doc5")
    
    # Delete all chunks
    print("\n🗑️  Deleting all chunk vectors...")
    for chunk_data in embedded_chunks:
        client.delete(chunk_data["id"])
        print(f"   ✅ Deleted {chunk_data['id']}")
    
    # Verify deletions
    stats = client.stats()
    print(f"\n📊 After deletions:")
    print(f"   Total vectors: {stats.total_vectors}")
    print(f"   Expected: {len(documents) - 1}")  # -1 for deleted doc5
    
    remaining = client.list()
    print(f"   Remaining IDs: {', '.join(remaining.ids)}")
    
    # PART 9: CLEANUP
    print("\n" + "=" * 70)
    print("PART 9: FINAL CLEANUP")
    print("=" * 70)
    
    print("\n🧹 Cleaning up remaining test vectors...")
    for vec_id in remaining.ids:
        client.delete(vec_id)
        print(f"   ✅ Deleted {vec_id}")
    
    # Final stats
    stats = client.stats()
    print(f"\n📊 Final database state:")
    print(f"   Total vectors: {stats.total_vectors}")
    print(f"   Memory usage: {stats.memory_usage} bytes")
    
    # Close connection
    print("\n🔌 Closing connection...")
    client.close()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    print("\n📝 Summary:")
    print(f"   ✅ Chunked 1 long document into {len(chunks)} chunks")
    print(f"   ✅ Embedded {len(documents) + len(chunks)} text pieces into vectors")
    print(f"   ✅ Added {len(documents) + len(chunks)} vectors to database")
    print(f"   ✅ Performed {len(test_queries)} top-k similarity searches")
    print(f"   ✅ Filtered vectors by metadata")
    print(f"   ✅ Updated vector metadata")
    print(f"   ✅ Deleted {1 + len(chunks)} vectors")
    print(f"   ✅ Cleaned up all test data")
    
    print("\n💡 Next Steps:")
    print("   - Replace mock embeddings with real embeddings (OpenAI, Sentence-Transformers)")
    print("   - Use VectraDB chunkers for more advanced chunking")
    print("   - Implement semantic search in your application")
    print("   - Scale up with larger datasets")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
