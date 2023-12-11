import time
import json
import requests
from pathlib import Path
from typing import List, Union, Iterator
from functools import partial

import gradio as gr
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceTextGenInference
from langchain.chains import ConversationChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Prerequisites:
# - start a backend server,
#   e.g. with any of the scripts in https://github.com/DFKI-NLP/perseus-textgen/blob/master/scripts

DEFAULT_API_ENDPOINT = "http://serv-3316.kl.dfki.de:5000"
DEFAULT_PARAMS = {
    "streaming": False,
    "do_sample": False,
    "max_new_tokens": 20,
    "repetition_penalty": None,
    "return_full_text": False,
    "seed": None,
    "stop_sequences": [],
    "temperature": 0.1,
    "top_k": None,
    "top_p": 0.95,
    "truncate": None,
    "typical_p": 0.95,
    "watermark": False,
}
TEMPLATES_FILE = Path(__file__).parent / "templates.json"
DEFAULT_TEMPLATE = "llama-2"


def get_info(endpoint: str) -> str:
    url = f"{endpoint}/info"
    r = requests.get(url=url)
    info = json.loads(r.text)
    return json.dumps(info, indent=2)


def user(user_message, history):
    return "", history + [[user_message, None]]


def bot(
    history: List[List[str]],
    endpoint: str, 
    parameter_str: str,
    template_str: str,
    system_prior: str,
    log_str: str,
    user_prefix_str: str,
    ai_prefix_str: str
) -> Iterator[List[List[str]]]:

    log = json.loads(log_str)
    parameters = json.loads(parameter_str)
    llm = HuggingFaceTextGenInference(
        inference_server_url=endpoint,
        **parameters
    )   
    template_str = template_str.replace("{system_prompt}", system_prior)
    prompt_template = PromptTemplate(
        input_variables=["history", "input"],
        template=template_str,
        template_format="jinja2"
    )
    memory = ConversationBufferMemory(human_prefix=user_prefix_str, ai_prefix=ai_prefix_str)
    for utterance_pair in history[:-1]:
        memory.save_context(
            inputs={"input": utterance_pair[0]},
            outputs={"output": utterance_pair[1]}
        )
    llm_chain = ConversationChain(
        prompt=prompt_template,
        llm=llm,
        verbose=True,
        memory=memory,
    )
    latest_user_message = history[-1][0].strip()
    log.append({"request": {"endpoint": endpoint, "prompt": latest_user_message, "history": memory.buffer, "details": True, **parameters}})
    history[-1][1] = ""
    log[-1]["response"] = ""
    try:
        if parameters.get("streaming", False):
            for response in llm_chain.predict(input=latest_user_message, callbacks=[StreamingStdOutCallbackHandler()]):
                history[-1][1] += response
                log[-1]["response"] += str(response)
                time.sleep(0.02)
                yield history, json.dumps(log, indent=2)
        else:
            response = llm_chain.predict(input=latest_user_message, callbacks=[StreamingStdOutCallbackHandler()])
            history[-1][1] = response
            log[-1]["response"] += str(response)
            yield history, json.dumps(log, indent=2)
    except Exception as e:
        log[-1]["response"] = str(e)
        del history[-1]
        yield history, json.dumps(log, indent=2)


def update_template_and_system_prior(template_key, template_str, system_prior, user_prefix_str, ai_prefix_str, templates):
    if template_key is None:
        return template_str, system_prior, user_prefix_str, ai_prefix_str
    new_template_data = templates[template_key]
    new_template_str = json.dumps(new_template_data["template"], indent=2)
    new_user_prefix_str = json.dumps(new_template_data["user_prefix"], indent=2)
    new_ai_prefix_str = json.dumps(new_template_data["ai_prefix"], indent=2)
    return new_template_str, new_template_data["system_prior"], new_user_prefix_str, new_ai_prefix_str


def start():
    # load templates from json file
    templates = json.load(open(TEMPLATES_FILE))
    # endpoint with info
    endpoint = gr.Textbox(lines=1, label="Address", value=DEFAULT_API_ENDPOINT)
    endpoint_info = gr.JSON(label="Endpoint info")
    # chatbot with parameters, prefixes, and system prompt
    parameters_str = gr.Code(label="Parameters", language="json", lines=10, value=json.dumps(DEFAULT_PARAMS, indent=2))
    template_str = gr.Code(
        label="Template (required keys: prompt, system_prompt, history, user_prompt, bot_prompt)",
        language="json",
        lines=6,
        value=json.dumps(templates[DEFAULT_TEMPLATE]["template"], indent=2),
    )
    user_prefix_str = gr.Code(label="User Prefix", language="json", lines=1, value=json.dumps(templates[DEFAULT_TEMPLATE]["user_prefix"], indent=2), interactive=True)
    ai_prefix_str = gr.Code(label="AI Prefix", language="json", lines=1, value=json.dumps(templates[DEFAULT_TEMPLATE]["ai_prefix"], indent=2), interactive=True)
    select_template = gr.Dropdown(
        label="Load Template and System Prior (overwrites existing values)",
        choices=list(templates),
    )
    system_prior = gr.Textbox(lines=5, label="System Prior", value="You are a helpful assistant.")
    chatbot = gr.Chatbot(label="Chat", show_copy_button=True)
    msg = gr.Textbox(label="User Prompt (hit Enter to send)")
    clear = gr.Button("Clear")
    undo = gr.Button("Undo")

    log_str = gr.Code(label="Requests and responses", language="json", lines=10, value="[]", interactive=False)

    with gr.Blocks(title="Simple TGI Frontend") as demo:
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tab("Endpoint"):
                    endpoint.render()
                with gr.Tab("Endpoint Info") as endpoint_info_tab:
                    endpoint_info.render()
                endpoint_info_tab.select(get_info, inputs=endpoint, outputs=endpoint_info, queue=False)

                with gr.Group():
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
                        inputs=[chatbot, endpoint, parameters_str, template_str, system_prior, log_str, user_prefix_str, ai_prefix_str],
                        outputs=[chatbot, log_str],
                    )

                    undo.render()
                    undo.click(lambda history: history[:-1], chatbot, chatbot, queue=False)
                    clear.render()
                    clear.click(lambda: None, None, chatbot, queue=False)

            with gr.Column(scale=1):
                parameters_str.render()
                with gr.Group():
                    template_str.render()
                    with gr.Row():
                        user_prefix_str.render()
                        ai_prefix_str.render()
                    system_prior.render()
                    select_template.render()
                    select_template.change(
                        partial(update_template_and_system_prior, templates=templates),
                        inputs=[select_template, template_str, system_prior, user_prefix_str, ai_prefix_str],
                        outputs=[template_str, system_prior, user_prefix_str, ai_prefix_str],
                        queue=False,
                    )

    demo.queue()
    demo.launch()


if __name__ == "__main__":
    start()
