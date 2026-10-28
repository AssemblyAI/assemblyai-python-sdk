# Python SDK API reference

The API reference uses public module docstrings and type annotations.

Use Python 3.12 to build the reference. Create and activate a virtual environment. Then install the SDK and the documentation tool:

```bash
python3.12 -m venv /tmp/assemblyai-sdk-docs
source /tmp/assemblyai-sdk-docs/bin/activate
python -m pip install -e . -r docs/requirements.txt
scripts/build_api_docs.sh
```

The script writes the HTML files to `docs/api/`.
