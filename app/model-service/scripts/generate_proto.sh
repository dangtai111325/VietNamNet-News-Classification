#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PROTO_DIR="$PROJECT_DIR/app/model-service/proto"
MODEL_DIR="$PROJECT_DIR/app/model-service/app/generated"
API_DIR="$PROJECT_DIR/app/api-service/app/generated"

python3 -m grpc_tools.protoc -I "$PROTO_DIR" --python_out="$MODEL_DIR" "$PROTO_DIR/classifier.proto"
cp "$MODEL_DIR/classifier_pb2.py" "$API_DIR/classifier_pb2.py"

