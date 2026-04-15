#!/bin/bash
source /environment.sh
dt-launchfile-init
echo "VEHICLE_NAME=$VEHICLE_NAME"


dt-exec rosrun my_package vehicle_client.py


dt-launchfile-join
