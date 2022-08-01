#!/bin/bash

set -e

exec yes | cp -rf /capture_temp/* /capture_volume &
exec python3 /app/main.py 