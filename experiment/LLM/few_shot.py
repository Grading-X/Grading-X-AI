import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"))

examples = [
    {
        'question': '같은 높이에서 떨어진 접시가 콘크리트 바닥에서는 깨지는데 솜 위에서는 깨지지 않는 현상과 같은 원리로 설명할 수 있는 현상을 세 가지 서술하시오',
        'answer': """안전을 위해 자동차에 에어백을 장착한다.
        야구공을 받을 때 손을 뒤로 빼면서 받는다.
        계단에서 뛰어내릴 때 무릎을 구부린다.""",
        'score': '10/10점',
        'reason': '예시로 제공한 세 가지 모두 충돌 시간을 길게하여 충격력이 작아지는 원리를 담고있다.',
    },
    {
        'question': '같은 높이에서 떨어진 접시가 콘크리트 바닥에서는 깨지는데 솜 위에서는 깨지지 않는 현상과 같은 원리로 설명할 수 있는 현상을 세 가지 서술하시오',
        'answer': """트램펄린의 경우 높이 뛰었다 착지해도 아프지않다.
        벽에 계란을 던진다.""",
        'score': '3/10점',
        'reason': '예시로 제공한 내용 중 트램펄린의 경우만 충돌 시간을 길게하여 충격력이 작아지는 원리를 담고있으며 세 가지 중 두 가지 현상만 서술하였다.',
    },
    {
        'question': '같은 높이에서 떨어진 접시가 콘크리트 바닥에서는 깨지는데 솜 위에서는 깨지지 않는 현상과 같은 원리로 설명할 수 있는 현상을 세 가지 서술하시오',
        'answer': """모르겠음""",
        'score': '0/10점',
        'reason': '문제에 대한 어떠한 적절한 예시도 제공하지 않음',
    },
]