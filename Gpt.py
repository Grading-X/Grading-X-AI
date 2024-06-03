import os
import re
import asyncio
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

async def llm_score(id, question, desired_answer, answer, weight, response_dic):
    llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"))
    final_prompt = pipeline_prompt.format(question=question,
                                          desired_answer=desired_answer,
                                          answer=answer)
    response = await llm.ainvoke(final_prompt)
    sentence = response.content
    match = re.search(r'점수:\s*(\d+)', sentence)
    response_dic[id] = float(match.group(1)) / 10 * weight # weight는 곱연산으로 처리

async def parallel_gpt(request_list, response_dic):
    await asyncio.gather(*[llm_score(value[0], value[1], value[2], value[3], value[4], response_dic) for value in request_list])

def main(request_list):
    response_dic = {}
    asyncio.run(parallel_gpt(request_list, response_dic))
    return response_dic