# NOTES:
# - model card: https://huggingface.co/google/flan-t5-large
# Works on A100-40GB, A100-PCI V100-32GB, RTX6000, RTX3090

srun -K \
--container-image=/netscratch/enroot/text-generation-inference_1.0.3.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME \
--container-workdir=$HOME \
-p RTX3090 \
--mem 16GB --gpus 1 \
--export MODEL_ID=google/flan-t5-large \
text-generation-launcher \
--port 5000

# HOW-TO ACCESS THE (EXECUTABLE) API DOCUMENTATION:
# First, you need to know the node your job is running on. Call this on the head node
# to get the list of your running jobs:
# squeue -u $USER
# This should give you a list of jobs, each with a node name in the "NODELIST(REASON)" column, e.g. "serv-3316".
# Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
# http://$NODE.kl.dfki.de:5000/docs
