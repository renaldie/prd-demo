import os
from typing import TypedDict
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, END, StateGraph
import streamlit as st

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
llm = AzureChatOpenAI(
    azure_endpoint="https://models.inference.ai.azure.com",
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2025-03-01-preview", 
    model_name="gpt-4.1-nano",
    temperature=1,
    api_key=GITHUB_TOKEN,
)

# State
class State(TypedDict):
    main_character_input: str
    location_input: str
    language: str
    story: str
    translated: str

# Chains
story_template = PromptTemplate.from_template(
    "Write a very very short story about a person named {main_character_input} in {location_input}."
)
story_chain = story_template | llm | StrOutputParser()

translate_template = PromptTemplate.from_template(
    "Translate the following story to {language}: \n\n{story}"
)
translate_chain = translate_template | llm | StrOutputParser()

# Nodes
def create_story(state):
    story = story_chain.invoke({
        "main_character_input": state["main_character_input"],
        "location_input": state["location_input"]
        })
    return {"story": story}

def translate_story(state):
    translated = translate_chain.invoke({
        "language": state["language"], 
        "story": state["story"]
        })
    return {"translated": translated}

# Build workflow
workflow = StateGraph(State)
workflow.add_node("create_story", create_story)
workflow.add_node("translate_story", translate_story)

workflow.add_edge(START, "create_story")
workflow.add_edge("create_story", "translate_story")
workflow.add_edge("translate_story", END)

app = workflow.compile(name="lullaby_maker")

def main():
    st.set_page_config(page_title="Children's Lullaby Maker",
                       layout="centered")
    st.title("Children's Lullaby Maker")
    st.header("Get Started...")

    location_input = st.text_input(label="Where is the story set?")
    main_character_input = st.text_input(label="Who is the main character?")
    language_input = st.text_input(label="Translate the story to which language?")

    submit_button = st.button("Submit")

    if location_input and main_character_input and language_input:
        if submit_button:
            with st.spinner("Generating lullaby..."):
                result = app.invoke({
                    "main_character_input": main_character_input,
                    "location_input": location_input,
                    "language": language_input
                })
                st.write("### Story")
                st.write(result["story"])
                st.write("### Translated Story to: " + f'{result["language"]}')
                st.write(result["translated"])

if __name__ == "__main__":
    main()