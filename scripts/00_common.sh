
_home=$(realpath $(dirname $0)/..)
CONFIG_DIR=$(realpath ${_home:-..}/scripts/00_configs)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

GPU=${GPU:-0}


OPTS=${OPTS:-""}
OPTS="${OPTS} --gpu ${GPU}"

LABEL_SHIFT=${LABEL_SHIFT:-0}
OPTS="${OPTS} --label_shift ${LABEL_SHIFT}"

error=0

cd $CONFIG_DIR

source 00_config.sh

source ${MODEL_OPTS}
source ${DATASET_OPTS}
