#!/bin/sh

# export NUMBA_NUM_THREADS=8

echo -n "runing on "
hostname

ulimit -v unlimited
ulimit -s unlimited
ulimit -u 2048


START_TIME=$(date +"%T")
echo "start program $@ at $START_TIME"

exec invoke "$@"

# memory profile
# python -m memory_profiler tasks.py "$@"

END_TIME=$(date +"%T")

echo "program $@ finished at $END_TIME"
