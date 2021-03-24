OPTIMIZER=${OPTIMIZER:-rmsprop}

BATCH_SIZE=${BATCH_SIZE:-24}
UPDATE_SIZE=${UPDATE_SIZE:-64}


LR_INIT=${LR_INIT:-1e-4}
LR_DECAY=${LR_DECAY:-1e-1}
LR_STEP=${LR_STEP:-20}
LR_TARGET=${LR_TARGET:-1e-6}
LR=${LR:-"-lr ${LR_INIT} -lrd ${LR_DECAY} -lrs ${LR_STEP} -lrt ${LR_TARGET}"}

DECAY=${DECAY:-5e-4}
EPOCHS=${EPOCHS:-60}

if [[ -z ${DATASET} ]]; then
	echo "DATASET ist not set!"
	error=1
fi

OUTPUT_SUFFIX=${OUTPUT_SUFFIX:-""}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"${_home:-..}/.results"}
_now=$(date +%Y-%m-%d-%H.%M.%S.%N)
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_PREFIX}/${DATASET}/${OPTIMIZER}${OUTPUT_SUFFIX}/${_now}}

# >>> Augmentations >>>
AUGMENTATIONS=${AUGMENTATIONS:-"random_crop random_flip color_jitter"}
OPTS="${OPTS} --augmentations ${AUGMENTATIONS}"
OPTS="${OPTS} --center_crop_on_val"
# >>>>>>>>>>>>>>>>>>>>>

OPTS="${OPTS} --epochs ${EPOCHS}"
OPTS="${OPTS} --optimizer ${OPTIMIZER}"
OPTS="${OPTS} --batch_size ${BATCH_SIZE}"
OPTS="${OPTS} --update_size ${UPDATE_SIZE}"
OPTS="${OPTS} --decay ${DECAY}"
OPTS="${OPTS} --output ${OUTPUT_DIR}"
OPTS="${OPTS} ${LR}"

