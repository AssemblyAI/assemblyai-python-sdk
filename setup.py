from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text()


def get_version() -> str:
    version = {}
    with open(Path(__file__).parent / "assemblyai" / "__version__.py") as f:
        exec(f.read(), version)
    return version["__version__"]


setup(
    name="assemblyai",
    version=get_version(),
    description="AssemblyAI Python SDK",
    author="AssemblyAI",
    author_email="engineering.sdk@assemblyai.com",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.19.0",
        "pydantic>=1.7.0,!=1.10.7",
        "typing-extensions>=3.7",
        "websockets>=11.0",
    ],
    extras_require={
        "extras": ["pyaudio>=0.2.13"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AssemblyAI/assemblyai-python-sdk",
    license="MIT License",
    license_files=["LICENSE"],
    python_requires=">=3.8",
    project_urls={
        "Code": "https://github.com/AssemblyAI/assemblyai-python-sdk",
        "Issues": "https://github.com/AssemblyAI/assemblyai-python-sdk/issues",
        "Documentation": "https://github.com/AssemblyAI/assemblyai-python-sdk/blob/master/README.md",
        "API Documentation": "https://www.assemblyai.com/docs/",
        "Website": "https://assemblyai.com/",
    },
)
