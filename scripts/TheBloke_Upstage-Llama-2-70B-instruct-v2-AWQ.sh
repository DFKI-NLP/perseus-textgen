# NOTES:
# - model card: https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ
# - required to mitigate cuda OOM on A100-40GB: `--max-batch-prefill-tokens 1024`
# - required because this model is quantized:`--quantize awq` (this, again, requires `text-generation-inference>=1.10`)

MODEL_ID=TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ
REVISION=6ffd8b58998a28840e40a97606074f80429ca5cf

srun -K \
--container-image=/netscratch/enroot/huggingface_text-generation-inference_1.1.0.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME \
--container-workdir=$HOME \
-p A100-PCI \
--mem 64GB \
--gpus 1 \
text-generation-launcher \
--model_id $MODEL_ID \
--revision $REVISION \
--quantize awq \
--max-batch-prefill-tokens 1024 \
--port 5000

# HOW-TO ACCESS THE (EXECUTABLE) API DOCUMENTATION:
# First, you need to know the node your job is running on. Call this on the head node
# to get the list of your running jobs:
# squeue -u $USER
# This should give you a list of jobs, each with a node name in the "NODELIST(REASON)" column, e.g. "serv-3316".
# Then, you can access the API documentation at the following endpoint (replace $NODE with the node name):
# http://$NODE.kl.dfki.de:5000/docs

