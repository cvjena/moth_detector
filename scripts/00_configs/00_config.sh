######################
#### Config files ####
######################

PYTHON_SETUP=${CONFIG_DIR:-"."}/00_python_setup.sh
SACRED_SETUP=${CONFIG_DIR:-"."}/00_sacred_setup.sh
DATASET_OPTS=${CONFIG_DIR:-"."}/10_dataset_opts.sh
MODEL_OPTS=${CONFIG_DIR:-"."}/11_model_opts.sh
TRAINING_OPTS=${CONFIG_DIR:-"."}/12_training_opts.sh
CLUSTER_SETUP=${CONFIG_DIR:-"."}/20_cluster_setup.sh
CLUSTER_TEARDOWN=${CONFIG_DIR:-"."}/30_cluster_teardown.sh

source ${PYTHON_SETUP}
