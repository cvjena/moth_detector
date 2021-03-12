############
### This config file defines different flavors, how to run python
############

_conda=${HOME}/.miniconda3
source ${_conda}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-chainer7cu11}


if [[ $GDB == "1" ]]; then
	echo "GDB debugging enabled!"

	PYTHON="gdb -ex run --args python"

elif [[ $MPI == "1" ]]; then
	echo "MPI execution enabled!"

	N_MPI=${N_MPI:-2}
	HOSTFILE=${HOSTFILE:-hosts.conf}

	# create hosts file with localhost only
	if [[ ! -f ${HOSTFILE} ]]; then
		echo "localhost slots=${N_MPI}" > ${HOSTFILE}
	fi
	ENV="-x PATH -x OMP_NUM_THREADS -x MONGODB_USER_NAME -x MONGODB_PASSWORD -x MONGODB_DB_NAME"
	PYTHON="orterun -n ${N_MPI} --hostfile ${HOSTFILE} --oversubscribe --bind-to socket ${ENV} python"
	OPTS="${OPTS} --mpi"

elif [[ $PROFILE == "1" ]]; then
	echo "Python profiler enabled!"

	PYTHON="python -m cProfile -o profile"

else
	PYTHON="python"
fi

RUN_SCRIPT=${RUN_SCRIPT:-"${_home}/main.py"}
