import json
from typing import List, Tuple, Dict

import gradio as gr
import requests

# Prerequisites:
# - start a backend server,
#   e.g. with any of the scripts in https://github.com/DFKI-NLP/perseus-textgen/blob/master/scripts

DEFAULT_API_ENDPOINT = "http://serv-3316.kl.dfki.de:5000"
DEFAULT_PARAMS = {
    "best_of": 1,
    "decoder_input_details": True,
    "details": True,
    "do_sample": True,
    "max_new_tokens": 20,
    "repetition_penalty": 1.03,
    "return_full_text": False,
    "seed": None,
    "stop": [
      "photographer"
    ],
    "temperature": 0.5,
    "top_k": 10,
    "top_n_tokens": 5,
    "top_p": 0.95,
    "truncate": None,
    "typical_p": 0.95,
    "watermark": True,
}

DEFAULT_FORMATTING = {
    "notes (this is not used)": "taken from https://huggingface.co/upstage/SOLAR-0-70b-16bit",
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


def assemble_inputs(
        system_prompt: str, chat_history: List[Tuple[str, str]], user_prompt: str, formatting: Dict[str, str]
) -> str:
    history = ""
    for hist_user_prompt, hist_bot_response in chat_history:
        if hist_user_prompt:
            history += formatting['user_message'].format(user_prompt=hist_user_prompt)
        if hist_bot_response:
            history += formatting['bot_message'].format(bot_response_without_prefix=hist_bot_response)
    if user_prompt:
        history += formatting['user_message'].format(user_prompt=user_prompt)
    inputs = formatting["inputs"].format(system_prompt=system_prompt, history=history) + formatting["bot_prefix"]
    return inputs


def generate(
        system_prompt: str,
        chat_history: List[Tuple[str, str]],
        user_prompt: str,
        endpoint: str,
        parameters: dict,
        formatting: Dict[str, str]
) -> Tuple[str, requests.Response]:
    inputs = assemble_inputs(
        system_prompt=system_prompt,
        chat_history=chat_history,
        user_prompt=user_prompt,
        formatting=formatting
    )
    url = f"{endpoint}/generate"
    headers = {'Content-Type': 'application/json'}
    params = {"inputs": inputs, "parameters": parameters}
    # TODO: implement streaming
    response = requests.post(url=url, json=params, headers=headers)
    generated_text = json.loads(response.text)["generated_text"]

    # strip bot prefix, if present
    if "bot_prefix" in formatting and generated_text.startswith(formatting["bot_prefix"]):
        generated_text = generated_text[len(formatting["bot_prefix"]):]

    return generated_text, response


def respond(user_message, chat_history, endpoint, params_str, formatting_str, system_prompt, log_str):
    bot_message, response = generate(
        system_prompt=system_prompt,
        chat_history=chat_history,
        user_prompt=user_message,
        endpoint=endpoint,
        parameters=json.loads(params_str),
        formatting=json.loads(formatting_str),
    )
    chat_history.append((user_message, bot_message))
    log_str += f"Request:\n{json.dumps(json.loads(response.request.body), indent=2)}\n\n"
    log_str += f"Response:\n{json.dumps(json.loads(response.text), indent=2)}\n\n"
    return "", chat_history, log_str


def start():
    # taken from https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
    # endpoint with info
    endpoint = gr.Textbox(lines=1, label="Endpoint", value=DEFAULT_API_ENDPOINT)
    endpoint_info = gr.Textbox(lines=1, label="Endpoint info")
    endpoint_info_btn = gr.Button(value="Get info")
    # chatbot with parameters, prefixes, and system prompt
    parameters_str = gr.Textbox(lines=10, label="Parameters", value=json.dumps(DEFAULT_PARAMS, indent=2))
    formatting_str = gr.Textbox(
        lines=5,
        label="Prefixes (required keys: inputs, user_message, bot_message)",
        value=json.dumps(DEFAULT_FORMATTING, indent=2),
    )
    system_prompt = gr.Textbox(lines=5, label="System Prompt", value="You are a helpful assistant.")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="User Prompt (hit Enter to send)")

    log_str = gr.Textbox(lines=10, label="Requests and responses")

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
                    # show the actual inputs send to the backend and its responses
                    log_str.render()

                msg.render()
                clear = gr.ClearButton([msg, chatbot])
                msg.submit(
                    respond,
                    inputs=[msg, chatbot, endpoint, parameters_str, formatting_str, system_prompt, log_str],
                    outputs=[msg, chatbot, log_str]
                )

            with gr.Column(scale=1):
                parameters_str.render()
                formatting_str.render()
                system_prompt.render()

    demo.launch()


if __name__ == "__main__":
    start()
