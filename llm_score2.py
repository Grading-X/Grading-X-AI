import os
import re
import asyncio
import time
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate
from dotenv import load_dotenv
load_dotenv()

full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """당신은 문제 채점을 수행해야 합니다. 주어진 문제와 모범 정답을 바탕으로 채점하세요."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """다음은 주어진 문제와 모범 정답입니다:

문제:{question}
모범정답:{desired_answer}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """이제 문제와 모범정답을 바탕으로 아래의 형식을 준수하여 마지막 답안에 대한 채점 결과를 이어서 작성하세요:

===형식===
답안: 형식용 답안1
점수: 7/10점
채점근거: 형식용 문장

답안: 형식용 답안2
점수: 5/10점
채점근거: 형식용 문장

답안: 형식용 답안3
점수: 10/10점
채점근거: 형식용 문장

===마지막 답안===
답안: {answer}"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

async def llm_score(id, question, desired_answer, answer, response_list):
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"))
    final_prompt = pipeline_prompt.format(
        question=question,
        desired_answer=desired_answer,
        answer=answer
    )
    response = await llm.ainvoke(final_prompt)
    sentence = response.content
    match = re.search(r'점수:\s*(\d+)', sentence)
    print(match.group())
    print('id', id, float(match.group(1)) / 10 * 50 )
    response_list.append((id, response.content))

async def main(response_list):
    await asyncio.gather(
        llm_score(1,
                    "열역학 제 1법칙을 설명하시오",
                  "어떤 계의 내부 에너지의 증가량은 계에 더해진 열 에너지에서 계가 외부에 해준 일을 뺀 양과 같다.",
                    '어떤 계의 내부 에너지의 증가량은 계에 더해진 열 에너지에서 계가 외부에 해준 일을 뺀 양과 같다.',
                  response_list),
        llm_score(2,
                  "열역학 제 1법칙을 설명하시오",
                  "어떤 계의 내부 에너지의 증가량은 계에 더해진 열 에너지에서 계가 외부에 해준 일을 뺀 양과 같다.",
                  '계의 내부 에너지는 보존된다.',
                  response_list),
        llm_score(3,
              "열역학 제 1법칙을 설명하시오",
                  "어떤 계의 내부 에너지의 증가량은 계에 더해진 열 에너지에서 계가 외부에 해준 일을 뺀 양과 같다.",
              '엔트로피가 일정하다',
                  response_list),
        llm_score(4,
                  "열역학 제 1법칙을 설명하시오",
                  "어떤 계의 내부 에너지의 증가량은 계에 더해진 열 에너지에서 계가 외부에 해준 일을 뺀 양과 같다.",
                  '어떤 계의 물체 A와 B가 열적 평형상태에 있고, B와 C가 열적 평형상태에 있으면 A와 C도 열평형상태에 있다.',
                  response_list)
        )

response_list = []
start = time.time()
asyncio.run(main(response_list))
print("걸린 시간: ", time.time() - start)

for response in response_list:
    print("응답 id", response[0], "응답 정보: ", response[1])