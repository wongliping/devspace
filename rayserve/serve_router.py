from starlette.requests import Request

import os
from ray import serve
from ray.serve.handle import DeploymentHandle
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage


@serve.deployment
class Multiplier:
    def __init__(self):
        pass

    def multiply(self, a: int, b: int) -> int:
        return a * b


@serve.deployment
class Router:
    def __init__(self, multiplier: DeploymentHandle):
        # Tool
        def multiply_tool(a: int, b: int) -> int:
            """Multiplies two integers using a remote multiplier handle.

            Args:
                a (int): The first integer to multiply.
                b (int): The second integer to multiply.

            Returns:
                int: The result of multiplying `a` and `b`.
            """
            return multiplier.multiply.remote(a, b).result()

        os.environ["OPENAI_API_KEY"] = "sk-svcacct-JTK_CWThqv-_-j4ERQRbMSAh92xncqOj18kIHpT_5f9t7M5UuWJAeN_awIj2jt78nfNl2T3BlbkFJB57kzP_F6VtYtqFt10lnEJZVnoVIZ_FrX_Bm5DAsJ63qcwL6QJZwRmL0EDW5_3MvVXJBQA"

        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
        llm_with_tools = llm.bind_tools([multiply_tool])

        # Node
        def tool_calling_llm(state: MessagesState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        # Build graph
        builder = StateGraph(MessagesState)
        builder.add_node("tool_calling_llm", tool_calling_llm)
        builder.add_node("tools", ToolNode([multiply_tool]))
        builder.add_edge(START, "tool_calling_llm")
        builder.add_conditional_edges(
            "tool_calling_llm",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", END)
        self.graph = builder.compile()
    
    async def __call__(self, http_request: Request) -> str:
        msg: str = await http_request.json()
        messages = [HumanMessage(content=msg)]
        messages = self.graph.invoke({"messages": messages})
        for m in messages['messages']:
            m.pretty_print()
        return messages['messages'][-1].content


multiply = Multiplier.bind()
app = Router.bind(multiply)