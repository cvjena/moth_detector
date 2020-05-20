PREPARE_TYPE=${PREPARE_TYPE:-model}
N_JOBS=${N_JOBS:-0}

if [[ -z ${_home} ]]; then
	echo "_home is not set!"
	error=1
fi

export DATA=${DATA:-${_home}/../dataset_info.yml}

DATASET=${DATASET:-AMMOD_MOTHS1}

OPTS="${OPTS} --prepare_type ${PREPARE_TYPE}"
OPTS="${OPTS} --n_jobs ${N_JOBS}"
