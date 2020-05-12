#!/usr/bin/env bash

_home=$(dirname $0)
_home=$(realpath $_home)
CONFIG_DIR=$(realpath ${_home:-.}/00_configs)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

GPU=${GPU:-0}


OPTS=${OPTS:-"--no_sacred"}
OPTS="${OPTS} --gpu ${GPU}"

LABEL_SHIFT=${LABEL_SHIFT:-0}
OPTS="${OPTS} --label_shift ${LABEL_SHIFT}"

error=0

cd $CONFIG_DIR

source 00_config.sh

source ${MODEL_OPTS}
source ${DATASET_OPTS}
source ${TRAINING_OPTS}
source ${CLUSTER_SETUP}

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
