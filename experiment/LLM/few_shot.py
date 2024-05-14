import os
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0.3, model_name="gpt-4o",
                     openai_api_key=os.environ.get("OPENAI_API_KEY"))