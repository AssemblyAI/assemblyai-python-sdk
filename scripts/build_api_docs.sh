#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS=error

pdoc -o docs/api \
  assemblyai \
  assemblyai.prerecorded.v2 \
  assemblyai.sync.v1 \
  assemblyai.dictation.v1 \
  assemblyai.llm_gateway.v1
