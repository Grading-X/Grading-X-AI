from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

sys_prompt: PromptTemplate = PromptTemplate(
    input_variables=["question1", "answer1", "question2", "answer2"],
    template="""당신은 시험 문제 출제자입니다. 
    다음과 같은 형식을 준수하여 문제, 답안을 생성해내야합니다. 
    문제: {question1}
    답안: {answer1}
    
    문제: {question2}
    답안: {answer2}

    위 형식을 준수하여 아래에서 주어질 문맥정보를 바탕으로 문제, 답안을 작성하세요."""
)
system_message_prompt = SystemMessagePromptTemplate(prompt=sys_prompt)

human_prompt: PromptTemplate = PromptTemplate(
    input_variables=["context"],
    template="문맥정보: {context}"
)
student_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, student_message_prompt]
)

formatted_prompt = chat_prompt.format(
    question1="프랑스의 수도는 어디인가요?",
    answer1="파리",
    question2="에펠탑은 어느 도시에 있나요?",
    answer2="파리",
    context="프랑스는 유럽에 위치한 나라로, 수도는 파리입니다."
)