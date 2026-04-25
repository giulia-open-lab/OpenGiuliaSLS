#!/bin/sh

# This file must be run from Giulia's root directory

current_date=$(date +"%Y%m%d")

# Without source code
docker build -t "iteamupv/giulia:nosource-$current_date-23.5.2-0-alpine" -t iteamupv/giulia:nosource-latest -f Dockerfile .

# With source code
docker build -t "iteamupv/giulia:$current_date-23.5.2-0-alpine" -t iteamupv/giulia:latest --build-arg INCLUDE_SOURCE=true -f Dockerfile .
