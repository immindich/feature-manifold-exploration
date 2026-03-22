#!/bin/bash

FLAGS="--min-count 1 --max-count 150 --sequences-per-count 80 --batch-size 10"

LAYERS_27B="5,15,25,35,45,55"
python collect_activations.py --model gemma-27b --output activations-27b.pt --layers $LAYERS_27B $FLAGS
python collect_activations.py --model gemma-27b --output activations-27b-target-only.pt --layers $LAYERS_27B $FLAGS --target-only

LAYERS_12B="4,12,20,28,36,44"
python collect_activations.py --model gemma-12b --output activations-12b.pt --layers $LAYERS_12B $FLAGS
python collect_activations.py --model gemma-12b --output activations-12b-target-only.pt --layers $LAYERS_12B $FLAGS --target-only
