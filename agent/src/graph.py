# © Copyright European Union - 2026

import os
from typing import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import json
import subprocess
from src.agents import (
    PlannedTask,
    ToolCall,
    instantiate_planner_agent,
    instantiate_code_writer_agent,
    instantiate_main_writer_agent,
    instantiate_main_fixer_agent,
    instantiate_assistant_agent
)
from src.utils import extract_json_from_reply

# Resolve paths relative to this file
_AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUN_DIR = os.path.join(_AGENT_DIR, "run")

# State definition
class ClusterState (TypedDict, total=False):
    question: Annotated[List[AnyMessage], add_messages]
    messages: Annotated[List[AnyMessage], add_messages]
    plan: List[Dict]
    code : str
    execution_result_error: str
    execution_result_out: str

class GraphAgent:
    # Agent nodes
    def user(self, state: ClusterState):
        state['messages'] = add_messages(state['messages'], state['question'])
        return {
            "question": [RemoveMessage(message.id) for message in state['question'][:-1]],
            "messages": state['messages']
        }
    
    def planner_node(self, state: ClusterState):
        history = []

        for message in state['messages']:
            if isinstance(message, HumanMessage):
                history.append(message)
            elif isinstance(message, AIMessage):
                if not message.tool_calls and message.content:
                    history.append(message)
        

        planner_agent = instantiate_planner_agent()
        planned_tasks = planner_agent.invoke(
            {
                # "chat_history": history,
                "user_message": state['question'][0].content
            }
        )
        plan_dict = json.loads(extract_json_from_reply(planned_tasks.content))
        state['plan'] = plan_dict

        return state
    
    def tools_writer_node(self, state: ClusterState):
        task_list = state['plan']

        # history = []

        code_writer_agent = instantiate_code_writer_agent()
        
        for task_id, task in enumerate(task_list):
            for tool_id, tool_call in enumerate(task.get('tool_calls')):
                prompt = """
                This is the user question:
                %s

                You are in the process of completing the following task:
                %s

                In particular, you are generating the following function:
                %s"""
                prompt = prompt % (state['question'][0].content, json.dumps(task), json.dumps(tool_call))

                task_code = code_writer_agent.invoke(
                    {
                        # "chat_history": history,
                        "user_message": prompt
                    }
                )
                clean_tool_code = task_code.content
                if '```python\n' in clean_tool_code:
                    clean_tool_code = clean_tool_code[10:len(clean_tool_code)-3]
                code_fp = os.path.join(_RUN_DIR, f"{task_id}.{tool_id}.{tool_call.get('tool_name')}.py")
                open(code_fp,"w").write(clean_tool_code)
                print(f"{tool_call.get('tool_name')} completed.")

                tool_call['code']= clean_tool_code

        return state
    
    def main_writer_node(self, state: ClusterState):
        plan = state['plan']
        history = state['question']

        code_writer_agent = instantiate_main_writer_agent()
        
        prompt = """
        Here is the plan with corresponding python functions:
        %s"""
        prompt = prompt % json.dumps(plan)

        task_code = code_writer_agent.invoke(
            {
                "chat_history": history,
                "user_message": prompt
            }
        )
        clean_code = task_code.content
        if '```python\n' in clean_code:
            clean_code = clean_code[10:len(clean_code)-3]
        code_fp = os.path.join(_RUN_DIR, "main.py")
        open(code_fp, "w").write(clean_code)
        state['code'] = clean_code
    
        return state
    
    def main_runner_node(self, state: ClusterState):
        main_script = os.path.join(_RUN_DIR, "main.py")
        
        process = subprocess.run([
            "python",
            main_script
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        state['execution_result_error'] = process.stderr
        state['execution_result_out'] = process.stdout
        
        return state
    
    def main_fixer_node(self, state: ClusterState):
        
        code_fixer_agent = instantiate_main_fixer_agent()
        
        prompt = """
        Here is the executed Python code:
        %s
        
        Here you can find the standard output of the execution:
        %s
        Here you can find the error generated:
        %s
        """
        prompt = prompt % (state['code'], state['execution_result_out'], state['execution_result_error'])

        task_code = code_fixer_agent.invoke(
            {
                "question": state['question'],
                "user_message": prompt
            }
        )
        clean_code = task_code.content
        if '```python\n' in clean_code:
            clean_code = clean_code[10:len(clean_code)-3]
        code_fp = os.path.join(_RUN_DIR, "main.py")
        open(code_fp, "w").write(clean_code)
        state['code'] = clean_code
        
        return state
    
    def response_node(self, state: ClusterState):
        assistant = instantiate_assistant_agent()
        response = assistant.invoke({
                "question": state['question'],
                "code": state['code'],
                "output": state['execution_result_out']
            }
        )
        return {"answer": response.content}

    # === Edges ===
    def should_fix_code(self, state: ClusterState):
        error_message = state.get('execution_result_error', "")
        if "Error" in str(error_message):
            return "fix_code"
        else:
            return "end"

    def build_graph(self) -> StateGraph:
        graph = StateGraph(ClusterState, input=ClusterState, output=ClusterState)
        # Define nodes
        graph.add_node("user", self.user)
        graph.add_node("planner", self.planner_node)
        graph.add_node("step_writer", self.tools_writer_node)
        graph.add_node("main_writer", self.main_writer_node)
        graph.add_node("main_runner", self.main_runner_node)
        graph.add_node("main_fixer", self.main_fixer_node)
        graph.add_node("response", self.response_node)
        # Define edges
        graph.add_edge(START, "user")
        graph.add_edge("user", "planner")
        graph.add_edge("planner", "step_writer")
        graph.add_edge("step_writer", "main_writer")
        graph.add_edge("main_writer", "main_runner")
        graph.add_conditional_edges(
            "main_runner",
            self.should_fix_code,
            {
                "fix_code": "main_fixer",
                "end": "response"
            }
        )
        graph.add_edge("main_fixer", "main_runner")
        graph.add_edge("response", END)

        memory = MemorySaver()
        graph = graph.compile(checkpointer=memory)
        return graph