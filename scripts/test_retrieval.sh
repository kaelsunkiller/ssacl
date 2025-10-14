#!/bin/bash
cuda=$1
cd "$(dirname "$0")/.." || exit 1
CUDA_VISIBLE_DEVICES=$cuda python -m DownStreams.Retrieval.Eval_Retrival