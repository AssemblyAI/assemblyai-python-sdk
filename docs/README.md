# Python SDK API reference

The API reference uses public module docstrings and type annotations. It is published at https://assemblyai.github.io/assemblyai-python-sdk/ on every push to `master`.

To build it locally, use Python 3.12. Create and activate a virtual environment. Then install the SDK and the documentation tool:

```bash
python3.12 -m venv /tmp/assemblyai-sdk-docs
source /tmp/assemblyai-sdk-docs/bin/activate
python -m pip install -e . -r docs/requirements.txt
scripts/build_api_docs.sh
```

The script writes the HTML files to `docs/api/`. Check that the public API is present:

```bash
python docs/check_api_docs.py
```
