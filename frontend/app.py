import json
from typing import List, Tuple, Dict, Iterator, Union

import gradio as gr
import requests
from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationResponse, TextGenerationStreamResponse

# Prerequisites:
# - start a backend server,
#   e.g. with any of the scripts in https://github.com/DFKI-NLP/perseus-textgen/blob/master/scripts

DEFAULT_API_ENDPOINT = "http://serv-3316.kl.dfki.de:5000"
DEFAULT_PARAMS = {
    # "details": False,  # this is set to True in the code
    "stream": False,
    # model: Optional[str] = None,  # we do not provide this functionality
    "do_sample": False,
    "max_new_tokens": 20,
    "best_of": None,
    "repetition_penalty": None,
    "return_full_text": False,
    "seed": None,
    "stop_sequences": None,
    "temperature": None,
    "top_k": None,
    "top_p": None,
    "truncate": None,
    "typical_p": None,
    "watermark": False,
    "decoder_input_details": False,
}
DEFAULT_TEMPLATE = {
    # "notes (this is not used)": "taken from https://huggingface.co/upstage/SOLAR-0-70b-16bit",
    "inputs": "### System: {system_prompt}\n{history}\n",
    "user_message": "### User: {user_prompt}\n",
    "bot_message": "### Assistant: {bot_response_without_prefix}\n",
    "bot_prefix": "### Assistant: ",
}


def get_info(endpoint: str) -> str:
    url = f"{endpoint}/info"
    r = requests.get(url=url)
    info = json.loads(r.text)
    return json.dumps(info, indent=2)


def assemble_prompt(
        system_prompt: str, history: List[Tuple[str, str]], template: Dict[str, str]
) -> str:
    history_str = ""
    for hist_user_prompt, hist_bot_response in history:
        if hist_user_prompt:
            history_str += template['user_message'].format(user_prompt=hist_user_prompt)
        if hist_bot_response:
            history_str += template['bot_message'].format(bot_response_without_prefix=hist_bot_response)

    inputs = template["inputs"].format(system_prompt=system_prompt, history=history_str) + template["bot_prefix"]
    return inputs


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(
        history, endpoint, parameters_str, formatting_str, system_prompt, log
) -> Iterator[Tuple[List[List[str]], Union[TextGenerationResponse, TextGenerationStreamResponse]]]:
    formatting = json.loads(formatting_str)
    prompt = assemble_prompt(
        system_prompt=system_prompt,
        history=history,
        template=formatting
    )
    parameters = json.loads(parameters_str)
    parameters["prompt"] = prompt
    parameters["details"] = True

    client = InferenceClient(model=endpoint)
    log.append({})
    if parameters.get("stream", False):
        history[-1][1] = ""
        for response in client.text_generation(**parameters):
            history[-1][1] += response.token.text
            log[-1] = {
                "request": parameters,
                "response": str(response),
            }
            yield history, log
    else:
        response = client.text_generation(**parameters)
        history[-1][1] = response.generated_text
        log[-1] = {
            "request": parameters,
            "response": str(response),
        }
        yield history, log


def start():
    # taken from https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
    # endpoint with info
    endpoint = gr.Textbox(lines=1, label="Endpoint", value=DEFAULT_API_ENDPOINT)
    endpoint_info = gr.JSON(label="Endpoint info")
    endpoint_info_btn = gr.Button(value="Get info")
    # chatbot with parameters, prefixes, and system prompt
    parameters_str = gr.Code(label="Parameters", language="json", lines=10, value=json.dumps(DEFAULT_PARAMS, indent=2))
    formatting_str = gr.Code(
        label="Template (required keys: inputs, user_message, bot_message)",
        language="json",
        lines=5,
        value=json.dumps(DEFAULT_TEMPLATE, indent=2),
    )
    system_prompt = gr.Textbox(lines=5, label="System Prompt", value="You are a helpful assistant.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="User Prompt (hit Enter to send)")
    clear = gr.Button("Clear")

    log = gr.JSON(label="Requests and responses", value=[])
    streaming = gr.Checkbox(label="Streaming")

    with gr.Blocks(title="Simple TGI Frontend") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                endpoint.render()
                endpoint_info.render()
                endpoint_info_btn.render()
                endpoint_info_btn.click(get_info, inputs=endpoint, outputs=endpoint_info)
                with gr.Tab("Dialog"):
                    chatbot.render()
                with gr.Tab("Request Log"):
                    log.render()

                msg.render()
                msg.submit(
                    user,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot],
                    queue=False
                ).then(
                    bot,
                    inputs=[chatbot, endpoint, parameters_str, formatting_str, system_prompt, log],
                    outputs=[chatbot, log],
                )
                clear.render()
                clear.click(lambda: None, None, chatbot, queue=False)

            with gr.Column(scale=1):
                parameters_str.render()
                formatting_str.render()
                system_prompt.render()
                streaming.render()

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    start()
