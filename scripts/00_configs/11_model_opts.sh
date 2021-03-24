FINAL_POOLING=${FINAL_POOLING:-g_avg}
WEIGHTS=${WEIGHTS:-"imagenet"}

# ssd frcnn
MODEL_TYPE=${MODEL_TYPE:-ssd}

case $MODEL_TYPE in
	"ssd" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=300
		else
			INPUT_SIZE=512
		fi
		;;
	"frcnn" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=300
		else
			INPUT_SIZE=600
		fi
		;;
esac

OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --pooling ${FINAL_POOLING}"
