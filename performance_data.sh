#!/bin/sh

python eval_counting.py --model gemma-27b --num-samples 2000 --max-length 2000 --max-count 400 --batch-size 8 --save gemma-27b-perf.json --plot gemma-27b-perf.png
python eval_counting.py --model gemma-12b --num-samples 2000 --max-length 2000 --max-count 400 --batch-size 8 --save gemma-12b-perf.json --plot gemma-12b-perf.png