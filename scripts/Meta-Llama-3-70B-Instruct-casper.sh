# NOTES:
# - model card: https://huggingface.co/casperhansen/llama-3-70b-instruct-awq
# Works on A100-80GB
# Doesn't work on  A100-GB

srun -K \
--container-image=/netscratch/enroot/text-generation-inference_2.0.5-dev0.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME,$HOME/.cache/huggingface:/root/.cache/huggingface     \
--container-workdir=$HOME       \
-p A100-80GB     \
--mem 64GB \
--gpus 1       \
text-generation-launcher \
--model-id casperhansen/llama-3-70b-instruct-awq \
--revision e578178ea893ca5e3326afd15da5aefa37e84d69 \
--quantize awq \
--port 5000

# HOW-TO ACCESS THE (EXECUTABLE) API DOCUMENTATION:
# First, you need to know the node your job is running on. Call this on the head node
# to get the list of your running jobs:
# squeue -u $USER
# This should give you a list of jobs, each with a node name in the "NODELIST(REASON)" column, e.g. "serv-3316".
# Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
# http://$NODE.kl.dfki.de:5000/docs


