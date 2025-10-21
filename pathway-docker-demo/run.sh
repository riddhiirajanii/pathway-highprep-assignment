#!/usr/bin/env bash
docker build -t pathway-demo:latest .
docker run --rm -v "$(pwd)/data":/app/data -v "$(pwd)/out":/app/out pathway-demo:latest
