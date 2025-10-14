#!/bin/bash
cuda=$1
cd "$(dirname "$0")/.." || exit 1
CUDA_VISIBLE_DEVICES=$cuda python -m DownStreams.Classification.ZeroShot_CLS.Zeroshot_Classification