from langchain.prompts.prompt import PromptTemplate

full_template = """{introduction}

{example}

{start}"""
full_prompt = PromptTemplate.from_template(full_template)

introduction_template = """당신은 문제 생성기입니다. 주어진 문맥정보를 바탕으로 예시의 형식을 준수하여 문제와 답안을 {N}개 생성하세요."""
introduction_prompt = PromptTemplate.from_template(introduction_template)