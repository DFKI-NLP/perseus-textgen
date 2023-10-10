srun -K \
--container-image=/netscratch/enroot/text-generation-inference_1.0.3.sqsh \
--container-mounts=/netscratch:/netscratch,/ds:/ds,/ds/models/llms/cache:/data,$HOME:$HOME \
--container-workdir=$HOME \
-p RTX3090 \
--mem 16GB --gpus 1 \
--export MODEL_ID=google/flan-t5-large \
text-generation-launcher \
--port 5000
