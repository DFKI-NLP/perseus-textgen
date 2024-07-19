# perseus-textgen
A repository for scripts to run awesomely large language models with text generation inference APIs and (chat) UIs


## Authentication
Certain models (e.g., Google-gemma, LLAMA3, etc.) require you to accept the license on the Hugging Face website. To download these models to your machine, you need an authentication token from [Hugging Face](https://huggingface.co/settings/tokens). To register your token on your machine, run the following command:

```sh
huggingface-cli login --token <yourToken>
