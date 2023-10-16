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
    "stream": True,
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
# taken from https://huggingface.co/upstage/SOLAR-0-70b-16bit
DEFAULT_TEMPLATE = {
    "prompt": "{system_prompt}{history}### Assistant: ",
    "system_prompt": "### System: {system_prior}\n",
    "user_prompt": "### User: {user_message}\n",
    "bot_prompt": "### Assistant: {bot_message}\n",
}


def get_info(endpoint: str) -> str:
    url = f"{endpoint}/info"
    r = requests.get(url=url)
    info = json.loads(r.text)
    return json.dumps(info, indent=2)


def assemble_prompt(
        system_prior: str, history: List[Tuple[str, str]], template: Dict[str, str]
) -> str:
    # assemble the system prompt
    system_prompt = template["system_prompt"].format(system_prior=system_prior)

    # assemble the history
    history_str = ""
    for user_message, bot_message in history:
        if user_message:
            history_str += template['user_prompt'].format(user_message=user_message)
        if bot_message:
            history_str += template['bot_prompt'].format(bot_message=bot_message)

    # create the final prompt
    prompt = template["prompt"].format(
        system_prompt=system_prompt,
        history=history_str,
    )

    return prompt


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(
        history, endpoint, parameters_str, template_str, system_prior, log_str
) -> Iterator[Tuple[List[List[str]], Union[TextGenerationResponse, TextGenerationStreamResponse]]]:
    log = json.loads(log_str)
    template = json.loads(template_str)
    prompt = assemble_prompt(
        system_prior=system_prior,
        history=history,
        template=template
    )
    parameters = json.loads(parameters_str)

    client = InferenceClient(model=endpoint)
    log.append({"request": {"endpoint": endpoint, "prompt": prompt, "details": True, **parameters}})
    try:
        if parameters.get("stream", False):
            history[-1][1] = ""
            for response in client.text_generation(prompt=prompt, details=True, **parameters):
                history[-1][1] += response.token.text
                log[-1]["response"] = str(response)
                yield history, json.dumps(log, indent=2)
        else:
            response = client.text_generation(prompt=prompt, details=True, **parameters)
            history[-1][1] = response.generated_text
            log[-1]["response"] = str(response)
            yield history, json.dumps(log, indent=2)
    except Exception as e:
        log[-1]["response"] = str(e)
        del history[-1]
        yield history, json.dumps(log, indent=2)


def start():
    # taken from https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
    # endpoint with info
    endpoint = gr.Textbox(lines=1, label="Address", value=DEFAULT_API_ENDPOINT)
    endpoint_info = gr.JSON(label="Endpoint info")
    # chatbot with parameters, prefixes, and system prompt
    parameters_str = gr.Code(label="Parameters", language="json", lines=10, value=json.dumps(DEFAULT_PARAMS, indent=2))
    template_str = gr.Code(
        label="Template (required keys: prompt, system_prompt, user_prompt, bot_prompt)",
        language="json",
        lines=6,
        value=json.dumps(DEFAULT_TEMPLATE, indent=2),
    )
    system_prior = gr.Textbox(lines=5, label="System Prior", value="You are a helpful assistant.")
    chatbot = gr.Chatbot(show_copy_button=True)
    msg = gr.Textbox(label="User Prompt (hit Enter to send)")
    clear = gr.Button("Clear")

    log_str = gr.Code(label="Requests and responses", language="json", lines=10, value="[]", interactive=False)

    with gr.Blocks(title="Simple TGI Frontend") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Endpoint"):
                    endpoint.render()
                with gr.Tab("Endpoint Info") as endpoint_info_tab:
                    endpoint_info.render()
                endpoint_info_tab.select(get_info, inputs=endpoint, outputs=endpoint_info, queue=False)
                # dummy element to separate above tabs from below tabs
                inbtw = gr.Button("Between", visible=False)
                with gr.Tab("Dialog"):
                    chatbot.render()
                with gr.Tab("Request Log"):
                    log_str.render()

                msg.render()
                msg.submit(
                    user,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot],
                    queue=False
                ).then(
                    bot,
                    inputs=[chatbot, endpoint, parameters_str, template_str, system_prior, log_str],
                    outputs=[chatbot, log_str],
                )
                clear.render()
                clear.click(lambda: None, None, chatbot, queue=False)

            with gr.Column(scale=1):
                parameters_str.render()
                template_str.render()
                system_prior.render()

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    start()
