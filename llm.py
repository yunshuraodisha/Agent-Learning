import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

class HelloAgentsLLM:
    def __init__(self,model: str = None,apikey:str = None,baseUrl:str = None,
    timeout:int = None,**kwargs):

        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apikey = apikey or os.getenv("LLM_API_KEY")
        self.baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        self.timeout = timeout

        if not all([self.model, self.apikey, self.baseUrl]):
            raise ValueError("Model, API key, and Base URL must be provided either as arguments or environment variables.") 

        self.client = OpenAI(api_key=self.apikey, base_url=self.baseUrl, timeout=self.timeout)

    def think(self,message:List[Dict[str, str]],temperature:float=0)-> str:
        """
        Returns a string that represents the LLM's thought process.
        """

        print("Thinking...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message,
                temperature=temperature    
            )

            print("模型成功接受请求并响应")

            collected_content=[]
            content = response.choices[0].message.content
            
            return content
        except Exception as e:
            print(f"调用模型时发生错误: {e}")
            return None

if __name__ == "__main__":

    try:
        llm = HelloAgentsLLM()
        message = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '天生我才必有用的下一句是什么？'}
        ]
        print('模型调用中……')
        response = llm.think(message)
        if response:
            print(f"模型响应: {response}")
                

    except ValueError as e:
        print(f"Error: {e}")

    