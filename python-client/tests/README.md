# Python Client Testing Guide

## Prerequisites

1. **VectraDB Server Running**: The tests require a running VectraDB server with gRPC enabled.

   Start the server from the project root (in WSL or Linux):
   ```bash
   cd src/server
   cargo run -- --enable-grpc -d 3 -D ./test_data
   ```

2. **Python Environment**: Python 3.8+ with required dependencies.

## Setup

Install the package in development mode with test dependencies:

```bash
cd python-client
pip install -e ".[dev]"
```

Or install dependencies manually:
```bash
pip install pytest pytest-asyncio grpcio grpcio-tools protobuf
```

## Generate gRPC Stubs

Before running tests or using the client, generate the gRPC stubs:

```bash
python generate_proto.py
```

This will create:
- `vectradb_client/vectradb_pb2.py` - Protocol buffer message definitions
- `vectradb_client/vectradb_pb2_grpc.py` - gRPC client/server stubs

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run with Verbose Output
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/test_sync_client.py -v
pytest tests/test_async_client.py -v
pytest tests/test_integration.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_sync_client.py::TestVectorOperations -v
pytest tests/test_async_client.py::TestAsyncSearch -v
```

### Run Specific Test
```bash
pytest tests/test_sync_client.py::TestVectorOperations::test_create_vector -v
```

### Run with Coverage
```bash
pytest tests/ --cov=vectradb_client --cov-report=html
```

View coverage report:
```bash
# On Windows
start htmlcov/index.html

# On Linux/WSL
xdg-open htmlcov/index.html
```

## Test Structure

### `conftest.py`
- Shared fixtures for all tests
- Test configuration (host, port, timeout)
- Cleanup utilities

### `test_sync_client.py`
- Tests for synchronous client (`VectraDBClient`)
- CRUD operations, search, list, stats
- Error handling

### `test_async_client.py`
- Tests for asynchronous client (`AsyncVectraDBClient`)
- Same coverage as sync tests but with async/await
- Batch operation tests

### `test_integration.py`
- End-to-end integration tests
- Requires running server
- Tests complete workflows

## Troubleshooting

### Server Not Running
```
Error: VectraDB server is not running or not accessible
```

**Solution**: Start the server before running tests:
```bash
cd src/server
cargo run -- --enable-grpc -d 3
```

### gRPC Stubs Not Found
```
ImportError: gRPC stubs not found. Please run 'python generate_proto.py' first
```

**Solution**: Generate the stubs:
```bash
python generate_proto.py
```

### Dimension Mismatch
```
VectraDBError: Vector dimension mismatch
```

**Solution**: Make sure the server is started with the correct dimension (3 for tests):
```bash
cargo run -- --enable-grpc -d 3
```

### Port Already in Use
```
Error: Address already in use
```

**Solution**: Check if another VectraDB instance is running:
```bash
# On Linux/WSL
lsof -i :50051

# On Windows (PowerShell)
netstat -ano | findstr :50051
```

Kill the process and restart the server.

## Test Data Cleanup

Tests use fixtures that automatically clean up test vectors. If tests fail and leave data:

```python
from vectradb_client import VectraDBClient

client = VectraDBClient()
vectors = client.list_vectors()
for vec in vectors:
    if vec.id.startswith('test_') or vec.id.startswith('async_'):
        client.delete_vector(vec.id)
client.close()
```

## Running Examples

After generating gRPC stubs, run the example scripts:

```bash
# Synchronous client
python examples/basic_sync.py

# Asynchronous client
python examples/basic_async.py

# Context manager examples
python examples/context_manager.py

# Error handling patterns
python examples/error_handling.py
```

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Python Client Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Rust
      uses: actions-rs/toolchain@v1
      with:
        toolchain: stable
    
    - name: Build and start VectraDB server
      run: |
        cd src/server
        cargo build --release
        ./../../target/release/vectradb-server --enable-grpc -d 3 &
        sleep 5
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        cd python-client
        pip install -e ".[dev]"
    
    - name: Generate gRPC stubs
      run: |
        cd python-client
        python generate_proto.py
    
    - name: Run tests
      run: |
        cd python-client
        pytest tests/ -v --cov=vectradb_client
```

## Performance Testing

For load testing, consider using `pytest-benchmark`:

```bash
pip install pytest-benchmark

# Run performance tests
pytest tests/ --benchmark-only
```

## Next Steps

1. ✅ Generate gRPC stubs: `python generate_proto.py`
2. ✅ Start VectraDB server with gRPC enabled
3. ✅ Run tests: `pytest tests/ -v`
4. ✅ Run examples to see the client in action
5. 📦 Install: `pip install -e .` to use in your own projects
