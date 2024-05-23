from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage

human_prompt = "New Question"
human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name="conversation"), human_message_template]
)

human_message = HumanMessage(content="Previous Question")
ai_message = AIMessage(
    content="""\
1. Response Example1

2. Response Example2 

3. Response Example3\
"""
)

formatted_prompt = chat_prompt.format(
    conversation=[human_message, ai_message]
)