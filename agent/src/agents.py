# © Copyright European Union - 2026

import os
import json
from typing import Dict, List, Annotated
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

LLM_API_KEY = os.environ["LLM_API_KEY"]
LLM_BASE_URL = os.environ["LLM_BASE_URL"]
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "llama-3.3-70b-instruct")
CODER_MODEL = os.environ.get("CODER_MODEL", "qwen-coder-2.5-instruct")


def get_client() -> OpenAI:
    """Create an OpenAI client using environment variables."""
    return OpenAI(
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL
    )


class ToolCall(BaseModel):
    tool_name: Annotated[str, ""]
    goal_of_the_tool_call: Annotated[str, "The information you want to get from the tool call."]


class PlannedTask(BaseModel):
    """The definition of one of the task to be executed."""
    task_title: Annotated[str, "The title of the task."]
    task_description: Annotated[str, "A description of the task to be executed."]
    tool_calls: Annotated[List[ToolCall], "The list of tool names to be called to accomplish the task."]


class Plan(BaseModel):
    """The plan to achieve the objective as a sequence of tasks."""
    plan: Annotated[List[PlannedTask], "The list of tasks to accomplish the objective."]


# === System Prompts ===

PLANNER_SYSTEM_PROMPT = """
You are a Task Planner for a specialized text clustering agent, whose goal is to clusterize a very large collection of texts.
The task you generate must be strictly guided by the usage rules outlined below.
1. Examine the User's query and determine the most appropriate sequence of task to be executed. For each task list the tools to be called in order to accomplish the task.
2. Construct a plan separating logical operations into tasks, and intend the tools as the steps to complete each task.
3. Provide short and meaningful names to tasks and tools.
4. In tool_calls, list the tools in the order in which they should be used, observing the constraints mentioned above.
5. Do not call the same tool multiple times on the same task.

Provide in output a JSON with the following structure, do not output anything else.
Before producing the output, check the JSON is valid. Correct it if not valid.

class ToolCall(BaseModel):
    tool_name: Annotated[str, ""]
    goal_of_the_tool_call: Annotated[str, "The information you want to get from the tool call."]

class PlannedTask (BaseModel):
    # The definition of one of the task to be executed.
    task_title: Annotated[str, "The title of the task."]
    task_description: Annotated[str, "A description of the task to be executed."]
    tool_calls: Annotated[List[ToolCall], "The list of tool names to be called to accomplish the task."]

Output:
    List[PlannedTask] # The list of task to achieve the objective
"""

CODE_WRITER_SYSTEM_PROMPT = """
You are a Python code writer agent. Your goal is to write a Python function to achieve the task provided by the user.
The code generation should observe the following points:
1. The code you generate must be a valid and complete Python function.
2. Write a single function including all the code needed to accomplish the task.
3. Include all the imports needed to run the function.
4. Comment the code so to make easily understendable.
5. Your complete response will be stored in a Python file and executed. Be sure you provide only valid Python in output.
6. If your task requires to translate or any other service requiring a Large Language Model, you can use an OpenAI-compatible endpoint. You can assume the api key is stored in an environment variable called "LLM_API_KEY" and the base url in "LLM_BASE_URL".
This is an example of code to connect the OpenAI client:
    from openai import OpenAI
    import os
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"]
    )
    chat_completion = client.chat.completions.create(
        model="llama-3.3-70b-instruct",
        messages=[...],
        stream=False,
        temperature=0.8
    )
7. Please consider providing helpful messages so that is easy to debug the code.
8. If you plan long operations, please provide monitoring of loops through tqdm.

You are provided with the definition of the plan you are working on, and the task within the plan for which you need to write the code.
Provide in output just the Python code, do not output anything else.
"""

MAIN_WRITER_SYSTEM_PROMPT = """
You are a Python code writer agent. Your goal is to write a single Python script, combining different functions previously built.
The code generation should observe the following points:
1. The code you generate must be a valid and complete Python script.
2. Include all the imports needed to run the script.
3. Comment the code so to make easily understendable.
4. Your complete response will be stored in a Python file and executed. Be sure you provide only valid Python in output.

You are provided with a complete plan to achieve a clustering task. Each task within the plan is made of tools, which are python functions detailed with the code.
Provide in output just the Python code, do not output anything else.
"""

MAIN_FIXER_SYSTEM_PROMPT = """
You are a Python code writer agent. Your goal is to fix the code of a Python script.
You will receive the error raised by the execution of the code as well as for the output provided by the execution.
Your task consists in fixing the provided code according to the error raised.

The code generation should observe the following points:
1. The code you generate must be a valid and complete Python script.
2. Include all the imports needed to run the script.
3. Comment the code so to make easily understendable.
4. Your complete response will be stored in a Python file and executed. Be sure you provide only valid Python in output.

Provide in output just the Python code, do not output anything else.
"""

ASSISTANT_SYSTEM_PROMPT = """
You are an assistant designed to answer user question.
A user asked to accomplish a task. Such task has been achieved through code: the code has been written through LLM and executed locally.
You will receive the code executed, together with the output. Provide the user with a message.
"""


# === Agent Functions ===

def invoke_llm(system_prompt: str, user_message: str, model: str = None, max_retries: int = 3) -> str:
    """Invoke the LLM using plain OpenAI SDK with retry logic."""
    client = get_client()
    if model is None:
        model = PLANNER_MODEL

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            continue


def instantiate_planner_agent():
    """Returns a callable planner agent using plain OpenAI SDK."""

    class PlannerAgent:
        def invoke(self, inputs: dict) -> "PlannerResponse":
            user_message = f"User question:\n\n {inputs['user_message']}"
            content = invoke_llm(PLANNER_SYSTEM_PROMPT, user_message, model=PLANNER_MODEL)
            return PlannerResponse(content=content)

    class PlannerResponse:
        def __init__(self, content):
            self.content = content

    return PlannerAgent()


def instantiate_code_writer_agent():
    """Returns a callable code writer agent using plain OpenAI SDK."""

    class CodeWriterAgent:
        def invoke(self, inputs: dict) -> "CodeResponse":
            user_message = f"User request:\n\n {inputs['user_message']}"
            content = invoke_llm(CODE_WRITER_SYSTEM_PROMPT, user_message, model=CODER_MODEL)
            return CodeResponse(content=content)

    class CodeResponse:
        def __init__(self, content):
            self.content = content

    return CodeWriterAgent()


def instantiate_main_writer_agent():
    """Returns a callable main writer agent using plain OpenAI SDK."""

    class MainWriterAgent:
        def invoke(self, inputs: dict) -> "MainResponse":
            chat_history = inputs.get("chat_history", [])
            history_text = ""
            if chat_history:
                history_text = "\n".join([
                    msg.content if hasattr(msg, 'content') else str(msg) 
                    for msg in chat_history
                ])
            system = MAIN_WRITER_SYSTEM_PROMPT + f"\n\n{history_text}"
            content = invoke_llm(system, inputs['user_message'], model=CODER_MODEL)
            return MainResponse(content=content)

    class MainResponse:
        def __init__(self, content):
            self.content = content

    return MainWriterAgent()


def instantiate_main_fixer_agent():
    """Returns a callable fixer agent using plain OpenAI SDK."""

    class FixerAgent:
        def invoke(self, inputs: dict) -> "FixerResponse":
            question = inputs.get("question", [])
            question_text = ""
            if question:
                question_text = "\n".join([
                    msg.content if hasattr(msg, 'content') else str(msg) 
                    for msg in question
                ])
            system = MAIN_FIXER_SYSTEM_PROMPT + f"\n\nHere is the user question: {question_text}"
            content = invoke_llm(system, inputs['user_message'], model=CODER_MODEL)
            return FixerResponse(content=content)

    class FixerResponse:
        def __init__(self, content):
            self.content = content

    return FixerAgent()


def instantiate_assistant_agent():
    """Returns a callable assistant agent using plain OpenAI SDK."""

    class AssistantAgent:
        def invoke(self, inputs: dict) -> "AssistantResponse":
            question = inputs.get("question", [])
            question_text = ""
            if question:
                question_text = "\n".join([
                    msg.content if hasattr(msg, 'content') else str(msg) 
                    for msg in question
                ])
            system = ASSISTANT_SYSTEM_PROMPT + f"\n\nHere is the user question: {question_text}"
            user_message = f"Here is the executed code:\n{inputs['code']}\n\nHere is the output:\n{inputs['output']}"
            content = invoke_llm(system, user_message, model=PLANNER_MODEL)
            return AssistantResponse(content=content)

    class AssistantResponse:
        def __init__(self, content):
            self.content = content

    return AssistantAgent()
