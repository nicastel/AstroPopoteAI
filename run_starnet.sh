#!/bin/bash
echo $@ > args.txt
cd /app
/opt/venv/bin/python3 mystarnet.py $@
