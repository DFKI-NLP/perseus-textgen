# perseus-textgen
A repository for scripts to run awesomely large language models with text generation inference APIs and (chat) UIs


## Authentication
Certain models (e.g., Google-gemma, LLAMA3, etc.) require you to accept the license on the Hugging Face website. To download these models to your machine, you need an authentication token from [Hugging Face](https://huggingface.co/settings/tokens). To register your token on your machine, run the following command:

```sh
huggingface-cli login --token <yourToken>
```
## Update the enroot image
Newer models sometimes require a never enroot image. To generate a new image adapt the following code-snippted. 

```sh
srun \
-p RTX3090 \
enroot import \
-o /netscratch/$USER/huggingface_text-generation-inference_1.1.0.sqsh \
docker://ghcr.io#huggingface/text-generation-inference:1.1.0
```

Check for available releases: [https://github.com/huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)
