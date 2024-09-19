from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import os
import time
from langchain_groq import ChatGroq
import subprocess


load_dotenv()


llm_groq = ChatGroq(

    model="llama-3.1-8b-instant",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)


# base_path = "C:\\Users\\ASUS\\OneDrive\\Desktop\\Python_projects"

# try:
#     # Open the base_path in a new VS Code window
#     subprocess.Popen(["code", base_path],  shell=True)
#     # subprocess.run(["code", base_path], check=True)
#     print(f"Project directory '{base_path}' opened in VS Code.")
# except Exception as e:
#     print(f"An error occurred: {e}")


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    try:
        # Limit to two sentences for brevity
        return wikipedia.run(query)
    except:
        return "I couldn't find any information on that."


def get_filename(query):
    """Extracts the filename from the answer."""
    msg = llm_groq.invoke(
        f"Extract the file or folder name name from following prompt,  return ONLY the file or folder name and nothing else. prompt: {query} ")

    return msg.content


def create_python_project(query):
    dir_name = get_filename(query)
    """Creates a new Python project directory."""
    base_path = "C:\\Users\\ASUS\\OneDrive\\Desktop\\Python_projects"
    project_path = os.path.join(base_path, dir_name)

    try:
        os.makedirs(project_path, exist_ok=True)
        # Open the new directory in a new VS Code window
        subprocess.Popen(["code", project_path],  shell=True)
        # subprocess.run(["code", project_path], check=True)
        return f"Project directory '{dir_name}' created successfully."
    except Exception as e:
        return f"An error occurred while creating the project directory: {e}"


def create_react_project(query):
    dir_name = get_filename(query)
    """Creates a new React project directory."""
    base_path = "C:\\Users\\ASUS\\OneDrive\\Desktop\\reactjs_prots"
    project_path = os.path.join(base_path, dir_name)

    try:
        os.makedirs(project_path, exist_ok=True)
        # Open the new directory in a new VS Code window
        subprocess.Popen(["code", project_path],  shell=True)
        subprocess.run(["code", project_path], check=True)
        return f"Project directory '{dir_name}' created successfully."
    except Exception as e:
        return f"An error occurred while creating the project directory: {e}"


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to know information about a  topic.",
    ),
    Tool(
        name="Create Python Project",
        func=create_python_project,
        description="Useful for when you need to create a new Python project directory.",
    ),
    Tool(
        name="Create React Project",
        func=create_react_project,
        description="Useful for when you need to create a new React or Web developemnt project directory.",
    )
]

prompt = hub.pull("hwchase17/structured-chat-agent")


# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Create a structured Chat Agent with Conversation Buffer Memory
# ConversationBufferMemory stores the conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

# create_structured_chat_agent initializes a chat agent designed to interact using a structured prompt and tools
# It combines the language model (llm), tools, and prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# AgentExecutor is responsible for managing the interaction between the user input, the agent, and the tools
# It also handles memory to ensure context is maintained throughout the conversation
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,  # Use the conversation memory to maintain context
    handle_parsing_errors=True,  # Handle any parsing errors gracefully
)

# Initial system message to set the context for the chat
# SystemMessage is used to define a message from the system to the agent, setting initial instructions or context
initial_message = "You are an AI assistant called Friday (Female). You can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))
# Chat Loop to interact with the user
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    # Add the user's message to the conversation memory
    memory.chat_memory.add_message(HumanMessage(content=user_input))

    # Invoke the agent with the user input and the current chat history
    response = agent_executor.invoke({"input": user_input})
    print("Bot:", response["output"])

    # Add the agent's response to the conversation memory
    memory.chat_memory.add_message(AIMessage(content=response["output"]))
