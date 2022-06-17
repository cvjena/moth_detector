#!/usr/bin/env bash
_home=${_home:-$(realpath $(dirname $0)/..)}

# Example calls:
#
# DATASET=AMMOD_MOTHS1 \
# LOAD=$(realpath  ~/Data/models/ssd/vgg16_extractor/ft_AMMOD_MOTHS1/model.npz) \
# 	./20_detect.sh
#
# DATASET=AMMOD_MOTHS1 \
# LOAD=$(realpath  ~/Data/models/ssd/vgg16_extractor/ft_AMMOD_MOTHS1/model.npz) \
# 	./20_detect.sh
# 	--vis_output $(realpath ../detections)

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

$PYTHON $RUN_SCRIPT detect \
	${DATA} \
	${DATASET} \
	GLOBAL \
	${OPTS} \
	$@
