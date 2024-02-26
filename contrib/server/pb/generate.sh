#!/bin/bash

# This script generates grpc/proto code from the .proto servce definition

set -e
set -x


# Generate C++ code

# sudo apt-get install --yes protobuf-compiler-grpc
PROTOC="../build/_deps/grpc-build/third_party/protobuf/protoc"
${PROTOC} -I . --grpc_out=. --plugin=protoc-gen-grpc=../build/_deps/grpc-build/grpc_cpp_plugin llm.proto
${PROTOC} -I . --cpp_out=. llm.proto


# Generate go code

# GOBIN=~/bin go install google.golang.org/protobuf/cmd/protoc-gen-go@v1.28
# GOBIN=~/bin go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.2
${PROTOC} -I . \
    --go_out=../clients/golang/pb --go_opt=paths=source_relative \
    --go-grpc_out=../clients/golang/pb --go-grpc_opt=paths=source_relative \
    llm.proto
