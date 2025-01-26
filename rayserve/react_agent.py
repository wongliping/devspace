from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


# from langchain_core.tools import tool

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
memory = MemorySaver()

system_prompt = "You are a helpful assistant tasked with performing arithmetic on a set of inputs."
graph = create_react_agent(llm, tools=tools, state_modifier=system_prompt, checkpointer=memory)

messages = [HumanMessage(content="Add 3 and 4. Multiply the output by 2. Divide the output by 5.")]

config = {"configurable": {"thread_id": "1"}}

# Stream mode
print("Streaming result:")
for event in graph.stream({"messages": messages}, config, stream_mode="values"):
    for m in event['messages']:
        m.pretty_print()

# Invoke mode
print("Invoke final result:")
messages = graph.invoke({"messages": messages}, config)
print(messages['messages'][-1].content)
