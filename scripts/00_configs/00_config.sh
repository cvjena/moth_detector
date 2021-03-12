######################
#### Config files ####
######################

PYTHON_SETUP=00_python_setup.sh
SACRED_SETUP=00_sacred_setup.sh
DATASET_OPTS=10_dataset_opts.sh
MODEL_OPTS=11_model_opts.sh
TRAINING_OPTS=12_training_opts.sh
CLUSTER_SETUP=20_cluster_setup.sh
CLUSTER_TEARDOWN=30_cluster_teardown.sh

source ${PYTHON_SETUP}
source ${SACRED_SETUP}
