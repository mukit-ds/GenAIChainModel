from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(

    template = 'Generate 5 interesting facts about {topic}',
    input_variables = ['topic']
)

llm = HuggingFaceEndpoint(
    repo_id = 'HuggingFaceH4/zephyr-7b-beta',
    task = 'text-generation',
    huggingfacehub_api_token="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"
)

model = ChatHuggingFace(llm = llm)
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'topic' : 'cricket'})

print(result)