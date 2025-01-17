#!/bin/bash

./build/bin/whisper-server \
  --model models/ggml-large-v3-turbo.bin \
  --language fi \
  --host 0.0.0.0 \
  --port 6543 \
  --ws-port 6544 \
  --no-timestamps \
  --max-context -1 \
  --prompt ""