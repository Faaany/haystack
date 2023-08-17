import os
import argparse

from datasets import load_dataset

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.agents import Tool
from haystack.nodes import PromptNode, PromptTemplate
from haystack.agents.memory import ConversationSummaryMemory
from haystack.agents import AgentStep, Agent
from haystack.agents.base import Agent, ToolsManager, Tool
from haystack.utils import print_answers

from prompt_templates import (
    agent_prompt_no_memory,
    agent_prompt_with_memory,
    summarizer_prompt_template,
    rag_qa_prompt_template,
    agent_prompt_no_memory2,
    agent_prompt_no_memory3,
    agent_prompt_original_no_memory,
    agent_prompt_original,
)

OPENAI_API_KEY=os.getenv("OPENAI_KEY")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--with-memory",
        default=False,
        action="store_true",
        help="whether the agent should has a memory or not",
    )
    parser.add_argument(
        "--use-openai-model",
        default=False,
        action="store_true",
        help="whether or not to use the OpenAI model for agent",
    )
    parser.add_argument(
        "--ipex",
        default=False,
        action="store_true",
        help="whether to use ipex for inference acceleration",
    )

    args = parser.parse_args()
    return args


def launch_document_store():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    document_store = InMemoryDocumentStore(use_bm25=True)
    document_store.write_documents(dataset)
    return document_store


def resolver_function(query, agent, agent_step):
    if with_memory:
        return {
            "query": query,
            "tool_names": agent.tm.get_tool_names(),
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
        "philschmid/bart-large-cnn-samsum",
        max_length=256,
        model_kwargs={"task_name": "text2text-generation"},
    )
    memory = ConversationSummaryMemory(
        memory_prompt_node, prompt_template="{chat_transcript}"
    )

    return memory


def create_agent(tools_manager, with_memory, use_openai_model, ipex):
    if use_openai_model:
        model_name = "gpt-3.5-turbo"
        agent_prompt_node = PromptNode(
            model_name,
            api_key=OPENAI_API_KEY,
            max_length=500,
            stop_words=["Observation:"],
            model_kwargs={"temperature": 0.5},
        )
    else:
        model_name = "lmsys/vicuna-13b-v1.3"
        agent_prompt_node = PromptNode(
            model_name,
            max_length=256,
            stop_words=["Observation:"],
            model_kwargs={
                "temperature": 0.5,
                "use_ipex": ipex,
                "torch_dtype": "torch.bfloat16",
            },
        )

    if with_memory:
        memory = create_memory()
        conversational_agent = Agent(
            agent_prompt_node,
            prompt_template=agent_prompt_original,
            prompt_parameters_resolver=resolver_function,
            memory=memory,
            tools_manager=tools_manager,
        )
    else:
        conversational_agent = Agent(
            agent_prompt_node,
            prompt_template=agent_prompt_original_no_memory,
            prompt_parameters_resolver=resolver_function,
            tools_manager=tools_manager,
        )

    return conversational_agent


def create_retriever_tool(document_store):
    retriever = BM25Retriever(document_store=document_store, top_k=3)
    retriever_pipeline = Pipeline()
    retriever_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])

    class MyRetrieverTool:
        def __init__(self, pipeline):
            self.pipeline = pipeline

        def __call__(self, query):
            documents = self.pipeline.run(query=query)["documents"]
            #joined = documents[0].content
            joined = "\n".join([doc.content for doc in documents])
            return joined

    callable_retriever = MyRetrieverTool(pipeline=retriever_pipeline)

    retriever_tool = Tool(
        name="retriever",
        pipeline_or_node=callable_retriever,
        description="useful when you need to find relevant paragraphs that might contain answers to a question",
        output_variable="documents",
    )

    return retriever_tool


def create_qa_tool(use_openai_model):
    if use_openai_model:
        prompt_node = PromptNode(
            model_name_or_path="text-davinci-003",
            api_key=OPENAI_API_KEY,
            default_prompt_template=rag_qa_prompt_template,
        )
    else:
        prompt_node = PromptNode(
            model_name_or_path="lmsys/vicuna-13b-v1.3",
            default_prompt_template=rag_qa_prompt_template,
        )

    rag_qa_tool = Tool(
        name="answer",
        pipeline_or_node=prompt_node,
        description="useful when you need to find an answer from some retrieved paragraphs",
        output_variable="answers",
    )

    return rag_qa_tool


def create_summarizer_tool(use_openai_model):
    if use_openai_model:
        prompt_node = PromptNode(
            model_name_or_path="text-davinci-003",
            api_key=OPENAI_API_KEY,
            default_prompt_template=summarizer_prompt_template,
        )
    else:
        prompt_node = PromptNode(
            model_name_or_path="lmsys/vicuna-13b-v1.3",
            default_prompt_template=rag_qa_prompt_template,
        )

    summarizer_tool = Tool(
        name="summarizer",
        pipeline_or_node=prompt_node,
        description="useful when you need to summarize a given paragraph",
        output_variable="answers",
    )

    return rag_qa_tool


if __name__ == "__main__":
    args = get_args()
    use_openai_model = args.use_openai_model
    with_memory = args.with_memory
    ipex = args.ipex 

    document_store = launch_document_store()
    retriever_tool = create_retriever_tool(document_store)
    rag_qa_tool = create_qa_tool(use_openai_model)
    summarizer_tool = create_summarizer_tool(use_openai_model)
    tools_manager = ToolsManager([retriever_tool, rag_qa_tool, summarizer_tool])

    agent = create_agent(tools_manager, with_memory, use_openai_model, ipex)

    agent.run("What did the Rhodes Statue look like?")