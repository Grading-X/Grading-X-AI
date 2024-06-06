from langchain.prompts.prompt import PromptTemplate
from langchain.prompts.pipeline import PipelinePromptTemplate

full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """당신은 문제 생성기입니다. 주어진 문맥정보를 바탕으로 예시의 형식을 준수하여 문제와 답안을 {N}개 생성하세요."""
introduction_prompt = PromptTemplate.from_template(introduction_template)

example_template = """다음은 문맥정보입니다:

{context}"""
example_prompt = PromptTemplate.from_template(example_template)

start_template = """이제 문맥정보를 바탕으로 아래의 형식을 준수하여 문제와 답변을 이어서 {N}개 생성하세요:

문제: {example_q1}
답안: {example_a1}

문제: {example_q2}
답안: {example_a2}

문제: {example_q3}
답안: {example_a3}"""
start_prompt = PromptTemplate.from_template(start_template)

input_prompts = [
    ("introduction", introduction_prompt),
    ("example", example_prompt),
    ("start", start_prompt),
]
pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_prompt, pipeline_prompts=input_prompts
)

formatted_prompt = pipeline_prompt.format(
        context = "문맥정보",
        example_q1="Question1",
        example_a1="Answer1",
        example_q2="Question2",
        example_a2="Answer2",
        example_q3="Question3",
        example_a3="Answer3",
        N=1,
    )