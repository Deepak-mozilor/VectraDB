from setuptools import setup, Extension
from pyo3_setuptools_rust import RustExtension

setup(
    name="vectradb-py",
    version="0.1.0",
    description="Python bindings for VectraDB - High-performance vector database",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="VectraDB Team",
    author_email="team@vectradb.dev",
    url="https://github.com/vectradb/vectradb",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    rust_extensions=[RustExtension("vectradb_py", "Cargo.toml", debug=False)],
    zip_safe=False,
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "black",
            "mypy",
            "flake8",
        ],
    },
)

