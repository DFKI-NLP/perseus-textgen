# NOTES:
# - model card: https://huggingface.co/TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ
# - required to mitigate cuda OOM on A100-40GB: `--max-batch-prefill-tokens 1024`
# - required because this model is quantized:`--quantize awq` (this, again, requires `text-generation-inference>=1.10`)

MODEL_ID=TheBloke/Upstage-Llama-2-70B-instruct-v2-AWQ
srun -K \
--container-image=/netscratch/enroot/huggingface_text-generation-inference_1.1.0.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME \
--container-workdir=$HOME \
-p A100-PCI \
--mem 64GB \
--gpus 1 \
--export MODEL_ID=$MODEL_ID \
text-generation-launcher \
--quantize awq \
--max-batch-prefill-tokens 1024 \
--port 5000

# Access the API at the following endpoint.
# TODO describe how to find the node your job is running on
# http://serv-33??.kl.dfki.de:5000
