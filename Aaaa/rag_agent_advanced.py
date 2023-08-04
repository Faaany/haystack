from datasets import load_dataset

from prompt_templates import (
    agent_prompt_no_memory,
    agent_prompt_with_memory,
    summarizer_prompt_template,
    rag_qa_prompt_template,
    agent_prompt_no_memory2,
    agent_prompt_no_memory3,
)

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.agents import Tool
from haystack.nodes import PromptNode, PromptTemplate
from haystack.agents.memory import ConversationSummaryMemory
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager
from haystack.utils import print_answers
import os

os.environ["TRANSFORMERS_CACHE"] = "/localdisk/fanlilin"

use_haystack_pipeline = False
with_memory = False
use_openai_model = True

openai_api_key = "xxxx"


def launch_document_store():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(dataset)

    return document_store


def resolver_function(query, agent, agent_step):
    if with_memory:
        return {
            "query": query,
            "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
            "transcript": agent_step.transcript,
            "memory": agent.memory.load(),
        }
    else:
        return {
            "query": query,
            "tool_names": agent.tm.get_tool_names(),
            "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
            "transcript": agent_step.transcript,
        }


def create_memory():
    memory_prompt_node = PromptNode(
        "philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
    )
    memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")

    return memory


def create_agent(tools_manager, with_memory, use_openai_model):
    if use_openai_model:
        model_name = "gpt-3.5-turbo"
        agent_prompt_node = PromptNode(
            model_name,
            api_key=openai_api_key,
            max_length=500,
            stop_words=["Observation:"],
            model_kwargs={"temperature": 0.5},
        )
    else:
        model_name = "lmsys/vicuna-13b-v1.3"
        agent_prompt_node = PromptNode(
            model_name, max_length=256, stop_words=["Observation:"], model_kwargs={"temperature": 0.5}
        )

    if with_memory:
        memory = create_memory()
        conversational_agent = Agent(
            agent_prompt_node,
            prompt_template=agent_prompt_with_memory,
            prompt_parameters_resolver=resolver_function,
            memory=memory,
            tools_manager=tools_manager,
        )
    else:
        conversational_agent = Agent(
            agent_prompt_node,
            prompt_template=agent_prompt_no_memory3,
            prompt_parameters_resolver=resolver_function,
            tools_manager=tools_manager,
        )

    return conversational_agent


if __name__ == "__main__":
    document_store = launch_document_store()

    retriever = BM25Retriever(document_store=document_store, top_k=3)

    if not use_haystack_pipeline:
        prompt_node_summarizer = PromptNode(
            model_name_or_path="text-davinci-003",
            api_key=openai_api_key,
            default_prompt_template=summarizer_prompt_template,
        )

        prompt_node_qa = PromptNode(
            model_name_or_path="text-davinci-003",
            api_key=openai_api_key,
            default_prompt_template=rag_qa_prompt_template,
        )

        # if use_openai_model:
        #     prompt_node = PromptNode(
        #         model_name_or_path="text-davinci-003", api_key=openai_api_key, default_prompt_template=summarizer_prompt_template
        #     )
        # else:
        #     prompt_node = PromptNode(
        #         model_name_or_path="lmsys/vicuna-13b-v1.3", default_prompt_template=qa_prompt_template
        #     )
        retriever_tool = Tool(
            name="retriever",
            pipeline_or_node=retriever,
            description="useful when you need to find relevant paragraphs that might contain answers to a question",
            output_variable="documents",
        )

        rag_qa_tool = Tool(
            name="question_answering",
            pipeline_or_node=prompt_node_qa,
            description="useful when you need to find an answer from some retrieved paragraphs",
            output_variable="answers",
        )

        summarizer_tool = Tool(
            name="summarizer",
            pipeline_or_node=prompt_node_summarizer,
            description="useful when you need to summarize a given paragraph",
            output_variable="answers",
        )

        tools_manager = ToolsManager([retriever_tool, rag_qa_tool, summarizer_tool])

    else:
        prompt_node = PromptNode(
            model_name_or_path="text-davinci-003",
            api_key=openai_api_key,
            default_prompt_template=summarizer_prompt_template,
            model_kwargs={"temperature": 0},
        )
        generative_pipeline = Pipeline()
        generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
        generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

        search_tool = Tool(
            name="seven_wonders_search",
            pipeline_or_node=generative_pipeline,
            description="useful for when you need to answer questions about the seven wonders of the world",
            output_variable="answers",
        )
        tools_manager = ToolsManager([search_tool])

    conversational_agent = create_agent(tools_manager, with_memory, use_openai_model)

    conversational_agent.run("What did the Rhodes Statue look like?")
