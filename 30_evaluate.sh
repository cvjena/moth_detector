#!/usr/bin/env bash

source 00_common.sh

if [[ -z ${LOAD} ]]; then
	echo "LOAD variable is not set!"
	error=1
fi

if [[ $error != 0 ]]; then
	exit $error
fi

BATCH_SIZE=${BATCH_SIZE:-4}

OPTS="${OPTS} --load ${LOAD}"
OPTS="${OPTS} --load_strict"
OPTS="${OPTS} --batch_size ${BATCH_SIZE}"

$PYTHON $RUN_SCRIPT evaluate \
	${DATA} \
	${DATASET} \
	GLOBAL \
	${OPTS} \
	$@
