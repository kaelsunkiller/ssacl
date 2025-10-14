#!/bin/bash
cuda=$1
cd "$(dirname "$0")/../DownStreams/Segmentation" || exit 1
bash test_single_gpu.sh $cuda