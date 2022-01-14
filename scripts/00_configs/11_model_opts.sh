FINAL_POOLING=${FINAL_POOLING:-g_avg}
WEIGHTS=${WEIGHTS:-"imagenet"}

# chainercv.SSD300 chainercv.FasterRCNNVGG16
MODEL_TYPE=${MODEL_TYPE:-chainercv.SSD300}

case $MODEL_TYPE in
	"chainercv.SSD300" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=300
		else
			INPUT_SIZE=512
		fi
		;;
	"chainercv.FasterRCNNVGG16" )
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
