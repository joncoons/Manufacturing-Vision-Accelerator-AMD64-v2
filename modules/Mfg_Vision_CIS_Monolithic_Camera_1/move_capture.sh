#!/bin/bash

set -e

exec yes | cp -rf /inference_temp/* /inference_volume &
exec python3 /app/main.py 