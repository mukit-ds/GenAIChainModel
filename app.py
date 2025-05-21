import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize prompt, LLM, model, parser, and chain once
prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    huggingfacehub_api_token="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"  # replace with env var in prod!
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
chain = prompt | model | parser

# Streamlit UI
st.title("🧠 Interesting Facts Generator")
st.write("Enter a topic and get 5 interesting facts!")

topic = st.text_input("Topic", value="cricket")

if st.button("Generate Facts"):
    if topic.strip() == "":
        st.warning("Please enter a topic.")
    else:
        with st.spinner("Generating..."):
            result = chain.invoke({'topic': topic})
        st.subheader(f"5 Facts about {topic}:")
        st.write(result)
