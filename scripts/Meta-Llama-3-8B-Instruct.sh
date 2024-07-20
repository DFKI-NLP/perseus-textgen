# NOTES:
# - model card: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# Works on A100-40GB
# Doesn't work on TBD

srun -K \
--container-image=/netscratch/enroot/text-generation-inference_2.0.5-dev0.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME,$HOME/.cache/huggingface:/root/.cache/huggingface     \
--container-workdir=$HOME       \
-p A100-40GB     \
--mem 64GB \
--gpus 1       \
text-generation-launcher \
--model-id meta-llama/Meta-Llama-3-8B-Instruct \
--revision e1945c40cd546c78e41f1151f4db032b271faeaa \
--port 5000

# HOW-TO ACCESS THE (EXECUTABLE) API DOCUMENTATION:
# First, you need to know the node your job is running on. Call this on the head node
# to get the list of your running jobs:
# squeue -u $USER
# This should give you a list of jobs, each with a node name in the "NODELIST(REASON)" column, e.g. "serv-3316".
# Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
# http://$NODE.kl.dfki.de:5000/docs

