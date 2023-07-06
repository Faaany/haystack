import warnings
import torch

from pathlib import Path

from haystack import Pipeline
from haystack import Document
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes import PromptTemplate
from transformers import (
    pipeline,
    AutoModel,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)


# name_or_path='mosaicml/mpt-7b-chat'
# name_or_path='decapoda-research/llama-7b-hf'
# name_or_path='EleutherAI/gpt-j-6b'
# pipe = pipeline('text-generation', model=name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
# #pipe = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased")
# print(pipe('Hello!'))
# print(pipe.tokenizer.model_max_length)


def load_model(name_or_path="mosaicml/mpt-7b-chat"):
    device = None
    device_map = "auto"
    model_dtype = torch.float32

    print(f"Loading HF Config...")
    from_pretrained_kwargs = {"use_auth_token": None, "trust_remote_code": True, "revision": None}
    config = AutoConfig.from_pretrained(name_or_path, **from_pretrained_kwargs)

    print(f"Loading HF Model...")
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, config=config, torch_dtype=model_dtype, device_map=device_map, **from_pretrained_kwargs
    )
    model.eval()

    print(f"Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **from_pretrained_kwargs)

    if tokenizer.pad_token_id is None:
        warnings.warn("pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


def get_generation_kwargs():
    # stop_tokens = ['<|endoftext|>', '<|im_end|>']
    # stop_token_ids = tokenizer.convert_tokens_to_ids(stop_tokens)
    stop_token_ids = [0, 50278]

    generate_kwargs = {
        "max_length": 2048,
        "temperature": 0.3,
        "top_p": 1.0,
        "top_k": 0,
        "use_cache": True,
        "do_sample": True,
        "eos_token_id": 0,
        "pad_token_id": 0,
    }

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    # streamer = TextStreamer(tokenizer,
    #                         skip_prompt=True,
    #                         skip_special_tokens=True)

    generate_kwargs = {
        **generate_kwargs,
        "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),
        # 'streamer':
        #     streamer
    }

    return generate_kwargs


class ChatFormatter:
    """A class for formatting the chat history.

    Args:
        system: The system prompt. If None, a default ChatML-formatted prompt is used.
        user: The user prompt. If None, a default ChatML value is used.
        assistant: The assistant prompt. If None, a default ChatML value is used.

    Attributes:
        system: The system prompt.
        user: The user prompt.
        assistant: The assistant prompt.
        response_prefix: The response prefix (anything before {} in the assistant format string)
    """

    def __init__(self, system: str, user: str, assistant: str) -> None:
        self.system = (
            system
            if system
            else "<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n"
        )
        self.user = user if user else "<|im_start|>user\n{}<|im_end|>\n"
        self.assistant = assistant if assistant else "<|im_start|>assistant\n{}<|im_end|>\n"
        self.response_prefix = self.assistant.split("{}")[0]


system_prompt = None
user_msg_fmt = None
assistant_msg_fmt = None

chat_format = ChatFormatter(system=system_prompt, user=user_msg_fmt, assistant=assistant_msg_fmt)


def format_history_str(history) -> str:
    text = "".join(
        ["\n".join([chat_format.user.format(item[0]), chat_format.assistant.format(item[1])]) for item in history[:-1]]
    )
    text += chat_format.user.format(history[-1][0])
    text += chat_format.response_prefix
    return text


def findLastIndex(str, x):
    # Traverse from right
    for i in range(len(str) - 1, -1, -1):
        if str[i] == x:
            return i

    return -1


# name_or_path = 'mosaicml/mpt-7b-chat'
# model, tokenizer = load_model(name_or_path)
# generate_kwargs = get_generation_kwargs(tokenizer)
# print(generate_kwargs)


# pipe = Pipeline()
# pipe.add_node(PromptNode(name_or_path, model_kwargs={"model":model, "tokenizer": tokenizer}), 'Prompter', inputs=['Query'])
# print("--------------")
# print(pipe)
# prompt_template = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n{query}"""

# result = pipe.run(query='Hello', params={
#     "Prompter": {"prompt_template": prompt_template},
#     } )


# name_or_path = 'mosaicml/mpt-7b-chat'
# model, tokenizer = load_model(name_or_path)


# pipe = Pipeline()
# pipe.add_node(PromptNode(name_or_path, model_kwargs={"model": model, "tokenizer": tokenizer}), 'Prompter', inputs=['Query'])
# print("--------------")

pipe = Pipeline.load_from_yaml(Path("multi_turn.yaml"))

prompt_template = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n{query}"""
generate_kwargs = get_generation_kwargs()
result = pipe.run(
    query="Hello", params={"Prompter": {"prompt_template": prompt_template, "generation_kwargs": generate_kwargs}}
)

print(result)


# generate_kwargs = get_generation_kwargs()

# pipeline = Pipeline.load_from_yaml(Path("multi_turn.yaml"))
# result = pipeline.run(query="Hello", params={
#     "Prompter": { "prompt_template": prompt_template, "generation_kwargs": generate_kwargs}
#     })
# print(result)

# generate_kwargs = get_generation_kwargs()
# model_name_or_path = "mosaicml/mpt-7b-chat"
# #model_name_or_path = 'decapoda-research/llama-7b-hf'
# prompt_template = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n{query}"""

# model_input_kwargs={'torch_dtype': torch.bfloat16}
# pipe = pipeline(
#             task='text-generation',  # task_name is used to determine the pipeline type
#             model=model_name_or_path,
#             device='cpu',
#             trust_remote_code=True,
#             #tokenizer=model_name_or_path,
#             model_kwargs=model_input_kwargs
#         )

# print(pipe('I want to buy a dress, can you help me with it?'))

# print(pipe.tokenizer)


# name_or_path = "mosaicml/mpt-7b-chat"
# generate_kwargs = get_generation_kwargs()

# #model, tokenizer = load_model(name_or_path)

# prompt_node = PromptNode("mosaicml/mpt-7b-chat",
#                         model_kwargs={"model": name_or_path, "tokenizer": name_or_path, "trust_remote_code": True}
#                         )

# pipe = Pipeline()
# pipe.add_node(prompt_node, 'Prompter', inputs=['Query'])

# prompt_template = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n{query}"""

# result = pipe.run(query='Hello', params={
#     "Prompter": {"prompt_template": prompt_template, "generation_kwargs": generate_kwargs},
#     } )

# print(result)


# name_or_path = 'mosaicml/mpt-7b-chat'
# model, tokenizer = load_model(name_or_path)
# generate_kwargs = get_generation_kwargs(tokenizer)
# print(generate_kwargs)

# pipe = Pipeline()
# pipe.add_node(PromptNode(name_or_path, model_kwargs={"model":model, "tokenizer": tokenizer}), 'Prompter', inputs=['Query'])
# print("--------------")
# print(pipe)
# prompt_template = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n{query}"""

# result = pipe.run(query='Hello', params={
#     "Prompter": {"prompt_template": prompt_template},
#     } )


# if __name__ == '__main__':

#     name_or_path = 'mosaicml/mpt-7b-chat'
#     model, tokenizer = load_model(name_or_path)

#     generate_kwargs = get_generation_kwargs()

#     prompt_node = PromptNode(name_or_path, debug=True, model_kwargs={"model":model, "tokenizer": tokenizer})

#     prompt_template = """<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n{query}"""

#     def generate_response(query, history):
#         history.append([query, ''])
#         conv = format_history_str(history)
#         results = prompt_node.run(query=conv, prompt_template=prompt_template, generation_kwargs=generate_kwargs)
#         results = results[0]['results'][0]
#         index = findLastIndex(results, '\n') +1
#         response = results[index:]
#         history[-1][-1] = response
#         return response, history

#     history = []
#     query1 = 'Write a welcome message to the user.'
#     response, history = generate_response(query1, history)
#     print('--------------------------')
#     print(response)

#     history = []
#     query2 = 'Hello'
#     response, history = generate_response(query2, history)
#     print('--------------------------')
#     print(response)

#     query3 = 'I want to buy a dress. Can you help me with it?'
#     response, history = generate_response(query3, history)
#     print('--------------------------')
#     print(response)

#     query4 = 'I want to have it in black.'
#     response, history = generate_response(query4, history)
#     print('--------------------------')
#     print(response)


# query = 'Write a welcome message to the user.'


# name_or_path = 'mosaicml/mpt-7b-chat'
# device = None
# device_map = 'auto'
# model_dtype = torch.float32
# stop_tokens = ['<|endoftext|>', '<|im_end|>']

# # pipe = pipeline('text-generation', model=name_or_path, tokenizer=name_or_path, trust_remote_code=True)
# # print(pipe.tokenizer.model_max_length)

# print(f'Loading HF Config...')
# from_pretrained_kwargs = {
#     'use_auth_token': None,
#     'trust_remote_code': True,
#     'revision': None
# }
# config = AutoConfig.from_pretrained(name_or_path,
#                                     **from_pretrained_kwargs)

# print(f'Loading HF Model...')
# model = AutoModelForCausalLM.from_pretrained(name_or_path,
#                                                 config=config,
#                                                 torch_dtype=model_dtype,
#                                                 device_map=device_map,
#                                                 **from_pretrained_kwargs)
# model.eval()

# print(f'Loading Tokenizer...')
# tokenizer = AutoTokenizer.from_pretrained(name_or_path,
#                                             **from_pretrained_kwargs)
# stop_token_ids = tokenizer.convert_tokens_to_ids(stop_tokens)

# if tokenizer.pad_token_id is None:
#     warnings.warn(
#         'pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id.'
#     )
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = 'left'

# generate_kwargs = {
#         'max_new_tokens': 2048,
#         'temperature': 0.3,
#         'top_p': 1.0,
#         'top_k': 0,
#         'use_cache': True,
#         'do_sample': True,
#         'eos_token_id': 0,
#         'pad_token_id': 0
#     }

# class StopOnTokens(StoppingCriteria):

#     def __call__(self, input_ids: torch.LongTensor,
#                     scores: torch.FloatTensor, **kwargs) -> bool:
#         for stop_id in stop_token_ids:
#             if input_ids[0][-1] == stop_id:
#                 return True
#         return False

# # streamer = TextStreamer(tokenizer,
# #                         skip_prompt=True,
# #                         skip_special_tokens=True)
# generate_kwargs = {
#     **generate_kwargs,
#     'stopping_criteria':
#         StoppingCriteriaList([StopOnTokens()]),
#     # 'streamer':
#     #     streamer
# }

# prompt_template_text = """<|im_start|>system
# A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>
# {query}"""
# prompt_template = PromptTemplate(prompt_template_text)


# query = 'Write a welcome message to the user.'

# prompt_node = PromptNode(name_or_path, debug=True, model_kwargs={"model":model, "tokenizer": tokenizer})
# results = prompt_node.run(query=query, prompt_template=prompt_template, generation_kwargs=generate_kwargs)
# print('%%%%%%%%%%%%%%%%%')
# print(results[0]['results'][0])

# class ChatFormatter:
#     """A class for formatting the chat history.

#     Args:
#         system: The system prompt. If None, a default ChatML-formatted prompt is used.
#         user: The user prompt. If None, a default ChatML value is used.
#         assistant: The assistant prompt. If None, a default ChatML value is used.

#     Attributes:
#         system: The system prompt.
#         user: The user prompt.
#         assistant: The assistant prompt.
#         response_prefix: The response prefix (anything before {} in the assistant format string)
#     """

#     def __init__(self, system: str, user: str, assistant: str) -> None:
#         self.system = system if system else '<|im_start|>system\nA conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.<|im_end|>\n'
#         self.user = user if user else '<|im_start|>user\n{}<|im_end|>\n'
#         self.assistant = assistant if assistant else '<|im_start|>assistant\n{}<|im_end|>\n'
#         self.response_prefix = self.assistant.split('{}')[0]

# system_prompt = None
# user_msg_fmt = None
# assistant_msg_fmt = None
# stop_tokens = "<|endoftext|> <|im_end|>"


# chat_format = ChatFormatter(system=system_prompt,
#                                 user=user_msg_fmt,
#                                 assistant=assistant_msg_fmt)

# query = 'I want to buy a dress. Can you help me with it?'

# history = [['Hello', 'Hi there! How can I help you today?']]

# history.append([query, ''])

# def format_history_str(self) -> str:
#     text = ''.join([
#         '\n'.join([
#             chat_format.user.format(item[0]),
#             chat_format.assistant.format(item[1]),
#         ]) for item in history[:-1]
#     ])
#     text += chat_format.user.format(history[-1][0])
#     text += chat_format.response_prefix
#     return text

# print("------------------------------")
# new_query = format_history_str(history)
# print(new_query)

# results = prompt_node.run(query=new_query, prompt_template=prompt_template, generation_kwargs=generate_kwargs)
# print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# response = results[0]['results'][0]
# print(response)

# def findLastIndex(str, x):

#     # Traverse from right
#     for i in range(len(str) - 1, -1,-1):
#         if (str[i] == x):
#             return i

#     return -1

# index = findLastIndex(response, '\n')
# print(response[index:])

# history[-1][-1] = output


# query = 'Hello'

# prompt_node = PromptNode(name_or_path, debug=True, model_kwargs={"model":model, "tokenizer": tokenizer})
# results = prompt_node.run(query=query, prompt_template=prompt_template, generation_kwargs=generate_kwargs)
# print(results)


# query = """Hello<|im_end|>

# <|im_start|>assistant
# Hi there! How can I help you today?<|im_end|>
# <|im_start|>user
# I want to buy a dress. Can you help me with it?
# """

# prompt_node = PromptNode(name_or_path, debug=True, model_kwargs={"model":model, "tokenizer": tokenizer})
# results = prompt_node.run(query=query, prompt_template=prompt_template, generation_kwargs=generate_kwargs)
# print(results)


# prompt_node = PromptNode(name_or_path, model_kwargs={"model":model, "tokenizer": tokenizer})

# pipeline = Pipeline()
# pipeline = Pipeline.load_from_yaml(Path("multi_turn.yaml"))
# pipeline.add_node(component=prompt_node, name='PromptNode', inputs=['Query'])

# query = 'Hello'

# output = pipeline.run(query=query, params={
#     "Prompter": {"prompt_template": prompt_template}
# })
# output = pipeline.run(query=query)
# output = pipeline.run(query=query, params={
#     "PromptNode": { "model":model, "tokenizer": tokenizer, "prompt_template": prompt_template},
#     })

# print(output)


# query = """Hello<|im_end|>

# <|im_start|>assistant
# Hi there! How can I help you today?<|im_end|>
# <|im_start|>user
# I want to buy a dress. Can you help me with it?
# """

# output = pipeline.run(query=query, params={
#     "PromptNode": { "prompt_template": prompt_template},
#     })

# print(output)


# query = """Hello<|im_end|>

# <|im_start|>assistant
# Hi there! How can I help you today?<|im_end|>
# <|im_start|>user
# I want to buy a dress. Can you help me with it?<|im_end|>

# <|im_start>user
# Sure, I can definitely help you with that! What sort of dress are you looking for?<|im_end|>
# <|im_start|>user
# I want to have it in black.
# """

# output = pipeline.run(query=query, params={
#     "PromptNode": { "prompt_template": prompt_template},
#     })

# print(output)
