srun -K \
--container-image=/netscratch/enroot/text-generation-inference_1.0.3.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME     \
--container-workdir=$HOME       \
-p A100-40GB     \
--mem 64GB \
--gpus 1       \
--export MODEL_ID=lmsys/vicuna-13b-v1.5 \
text-generation-launcher \
--port 5000

# Access the API at the following endpoint.
# TODO describe how to find the node your job is running on
# http://serv-33??.kl.dfki.de:5000

