FINAL_POOLING=${FINAL_POOLING:-g_avg}

# resnet inception inception_tf [vgg]
MODEL_TYPE=${MODEL_TYPE:-inception_imagenet}

case $MODEL_TYPE in
	"inception" | "inception_imagenet" | "inception_inat" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=299
		else
			INPUT_SIZE=427
		fi
		;;
	"resnet" )
		if [[ ${BIG:-0} == 0 ]]; then
			INPUT_SIZE=224
		else
			INPUT_SIZE=448
		fi
		;;
	"efficientnet" )
		INPUT_SIZE=380
		;;
esac

OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --input_size ${INPUT_SIZE}"
OPTS="${OPTS} --pooling ${FINAL_POOLING}"
