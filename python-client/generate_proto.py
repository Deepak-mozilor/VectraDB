#!/usr/bin/env python3
"""
Script to generate Python gRPC code from proto files.
Run this script to regenerate the gRPC stubs after modifying the proto file.
"""

import os
import sys
from grpc_tools import protoc

def main():
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proto_dir = os.path.join(project_root, "proto")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectradb_client")
    
    print(f"Proto directory: {proto_dir}")
    print(f"Output directory: {output_dir}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate Python gRPC code
    proto_file = os.path.join(proto_dir, "vectradb.proto")
    
    if not os.path.exists(proto_file):
        print(f"Error: Proto file not found at {proto_file}")
        sys.exit(1)
    
    print(f"Generating gRPC code from {proto_file}...")
    
    result = protoc.main([
        'grpc_tools.protoc',
        f'-I{proto_dir}',
        f'--python_out={output_dir}',
        f'--grpc_python_out={output_dir}',
        proto_file,
    ])
    
    if result == 0:
        print("✓ Successfully generated gRPC stubs!")
        print(f"  - {os.path.join(output_dir, 'vectradb_pb2.py')}")
        print(f"  - {os.path.join(output_dir, 'vectradb_pb2_grpc.py')}")
        
        # Fix imports in generated gRPC file
        grpc_file = os.path.join(output_dir, 'vectradb_pb2_grpc.py')
        with open(grpc_file, 'r') as f:
            content = f.read()
        
        # Convert absolute import to relative import
        content = content.replace('import vectradb_pb2 as vectradb__pb2', 
                                 'from . import vectradb_pb2 as vectradb__pb2')
        
        with open(grpc_file, 'w') as f:
            f.write(content)
        
        print("✓ Fixed imports in generated files")
    else:
        print(f"✗ Failed to generate gRPC stubs (exit code: {result})")
        sys.exit(result)

if __name__ == "__main__":
    main()
