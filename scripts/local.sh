# usage: local.sh <model_id_or_local_path>
# e.g. local.sh /ds/models/llms/llama_hf/7b

MODEL_ID=$1

srun -K \
--container-image=/netscratch/enroot/huggingface_text-generation-inference_1.1.0.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME     \
--container-workdir=$HOME       \
-p A100-40GB     \
--mem 64GB \
--gpus 1       \
text-generation-launcher \
--model-id $MODEL_ID \
--port 5000

# HOW-TO ACCESS THE (EXECUTABLE) API DOCUMENTATION:
# First, you need to know the node your job is running on. Call this on the head node
# to get the list of your running jobs:
# squeue -u $USER
# This should give you a list of jobs, each with a node name in the "NODELIST(REASON)" column, e.g. "serv-3316".
# Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
# http://$NODE.kl.dfki.de:5000/docs


