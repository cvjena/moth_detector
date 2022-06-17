#!/usr/bin/env bash
_home=${_home:-$(realpath $(dirname $0)/..)}

source 00_common.sh

source ${TRAINING_OPTS}
source ${SACRED_SETUP}
source ${CLUSTER_SETUP}

OPTS="${OPTS} --no_sacred"

if [[ $error != 0 ]]; then
	exit $error
fi

$PYTHON $RUN_SCRIPT train \
	${DATA} \
	${DATASET} \
	GLOBAL \
	${OPTS} \
	$@

source ${CLUSTER_TEARDOWN}
