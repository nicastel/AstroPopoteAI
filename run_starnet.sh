#!/bin/bash
echo $@ > args.txt
cd /app
export TF_USE_LEGACY_KERAS=1
/opt/venv/bin/python3 mystarnet.py $@
