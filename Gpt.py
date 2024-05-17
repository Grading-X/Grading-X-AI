import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
import re

examples = [
    {
        'question': '같은 높이에서 떨어진 접시가 콘크리트 바닥에서는 깨지는데 솜 위에서는 깨지지 않는 현상과 같은 원리로 설명할 수 있는 현상을 세 가지 서술하시오',
        'answer': """안전을 위해 자동차에 에어백을 장착한다.
        야구공을 받을 때 손을 뒤로 빼면서 받는다.
        계단에서 뛰어내릴 때 무릎을 구부린다.""",
        'weight': '10',
        'score': '10/10',
        'reason': '예시로 제공한 세 가지 모두 충돌 시간을 길게하여 충격력이 작아지는 원리를 담고있다.',
    },
    {
        'question': '같은 높이에서 떨어진 접시가 콘크리트 바닥에서는 깨지는데 솜 위에서는 깨지지 않는 현상과 같은 원리로 설명할 수 있는 현상을 세 가지 서술하시오',
        'answer': """트램펄린의 경우 높이 뛰었다 착지해도 아프지않다.
        벽에 계란을 던진다.""",
        'weight': '100',
        'score': '30/100',
        'reason': '예시로 제공한 내용 중 트램펄린의 경우만 충돌 시간을 길게하여 충격력이 작아지는 원리를 담고있으며 세 가지 중 두 가지 현상만 서술하였다.',
    },
    {
        'question': '같은 높이에서 떨어진 접시가 콘크리트 바닥에서는 깨지는데 솜 위에서는 깨지지 않는 현상과 같은 원리로 설명할 수 있는 현상을 세 가지 서술하시오',
        'answer': """모르겠음""",
        'weight': '50',
        'score': '0/50',
        'reason': '문제에 대한 어떠한 적절한 예시도 제공하지 않음',
    },
]

example_prompt = PromptTemplate(
    input_variables=['question', 'answer', 'weight', 'score', 'reason'] , template="문제:{question}\n학생답안:{answer}\n배점:{weight}\n점수:{score}\n채점근거:{reason}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="문제:{question}\n학생답안:{answer}\n배점:{weight}",
    input_variables=['question', 'answer', 'weight']
)

async def llm_score(id, question, answer, weight, response_dic):
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"))
    final_prompt = prompt.format(question=question, answer=answer, weight=weight)
    response = await llm.ainvoke(final_prompt)
    sentence = response.content
    index = sentence.find('점수')
    number = re.search(r'\d+', sentence[index:])
    response_dic[id] = float(number.group())

async def parallel_gpt(request_list, response_dic):
    await asyncio.gather(*[llm_score(value[0], value[1], value[2], value[3], response_dic) for value in request_list])

def main(request_list):
    response_dic = {}
    asyncio.run(parallel_gpt(request_list, response_dic))
    return response_dic