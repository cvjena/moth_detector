#!/usr/bin/env bash

source 00_common.sh

if [[ -z ${LOAD} ]]; then
	echo "LOAD variable is not set!"
	error=1
fi

if [[ $error != 0 ]]; then
	exit $error
fi

OPTS="${OPTS} --load ${LOAD}"
OPTS="${OPTS} --load_strict"

$PYTHON $RUN_SCRIPT detect \
	${DATA} \
	${DATASET} \
	GLOBAL \
	${OPTS} \
	$@
