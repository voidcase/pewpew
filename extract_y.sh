#!/bin/bash
(
	source /usr/local/dials-v1-11-4/dials_env.sh 
	distl.signal_strength $1 | grep "Spot Total" | awk '{print $NF}'
)
