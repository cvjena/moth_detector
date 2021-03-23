#!/usr/bin/env bash


if [[ -z ${MODELS} ]]; then
	echo "MODELS variable is not set!"
	exit 1
fi

export OPTS="--eval_methods voc --voc_thresh 0.5 0.75"

for model in $MODELS; do
	output="$(dirname ${model})/evaluation"
	LOAD=$model ./30_evaluate.sh $@ |& tee $output
done
