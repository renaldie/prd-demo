import os
import tempfile
import json
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from typing import List, Dict, Any

from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from markitdown import MarkItDown

# Set page title and configure the page
st.set_page_config(page_title="Smart PRD", layout="wide")
st.title("Smart-PRD")
st.write("AI-Powered Requirement Management Engineer.")
st.markdown("---")

# Load environment variables
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN", "")
OPEN_AI_API_KEY = os.environ.get("OPEN_AI_API_KEY") or st.secrets.get("OPEN_AI_API", "")

# Initialize LLM
# @st.cache_resource
# def get_llm():
#     return ChatOpenAI(
#         model_name="gpt-4.1-nano",
#         temperature=1,
#         openai_api_key=OPEN_AI_API_KEY,
#     )

@st.cache_resource
def get_llm():
    return AzureChatOpenAI(
    azure_endpoint="https://models.inference.ai.azure.com",
    azure_deployment="gpt-4.1-nano",
    openai_api_version="2025-03-01-preview", 
    model_name="gpt-4.1-nano",
    temperature=1,
    api_key=GITHUB_TOKEN,
    )

# Pydantic models
class Metadata(BaseModel):
    """Metadata of the document."""
    document_title: str = Field(description="The name of the document", default="")
    document_date: str = Field(description="The date of the document", default="")
    document_status: str = Field(description="The status of the document", default="")
    document_author: str = Field(description="The author of the document", default="")

class Person(BaseModel):
    """Person in the document beside the author."""
    name: str = Field(description="The name of the person beside the author", default="")
    role: str = Field(description="The role of the person beside the author", default="")
    email: str = Field(description="The email of the person beside the author", default="")

class Problem(BaseModel):
    """Overview of the meeting minutes"""
    vision_opportunity: str = Field(description="The Vision & Opportunity of the document", default="")
    target_use_case: str = Field(description="The Target Use Case of the document", default="")

class Solution(BaseModel):
    """Solution to the problem"""
    goals: str = Field(description="The goals of the solution", default="")
    conceptual_model: str = Field(description="The conceptual model of the solution", default="")
    requirements: List[str] = Field(description="The requirements of the solution", default="")

class Summary(BaseModel):
    metadata: Metadata = Field(description="The metadata of the document")
    person: Person = Field(description="The person in the document")
    problem: Problem = Field(description="The problem in the document")
    solution: Solution = Field(description="The solution in the document")

summary_parser = PydanticOutputParser(pydantic_object=Summary)

def extract_prd(prd):
    prd_template = """"
    You are an expert in analyzing Product Requirements Documents (PRDs).
    Your task is to extract software requirements from the provided PRD text.
    
    PRD text: {prd}
    
    For each PRD text, provide the following information:
    {format_instructions}
    
    If a section doesn't contain any requirements, output nothing for that section.
    """

    prd_prompt_template = PromptTemplate(
        input_variables=["prd"],
        template=prd_template,
        partial_variables={"format_instructions": summary_parser.get_format_instructions()},
    )

    LLM = get_llm()
    chain = prd_prompt_template | LLM | summary_parser
    output = chain.invoke(input={"prd": prd})
    return output

def convert_to_md(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    md = MarkItDown()
    result = md.convert(tmp_file_path)
    return result.text_content

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility='hidden')

if uploaded_file is not None:
    if st.button("Extract PRD Metadata"):
        with st.spinner("Processing the PDF..."):
            try:
                # Convert PDF to text
                text = convert_to_md(uploaded_file)
                summary = extract_prd(text)

                st.markdown("---")
                st.markdown("## Extracted Text")
                
                # Display the results
                ## Metadata
                st.markdown("### Metadata")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Document Title**")
                    st.write(summary.metadata.document_title)
                    
                    st.markdown("**Document Date**")
                    st.write(summary.metadata.document_date)
                
                with col2:
                    st.markdown("**Document Status**")
                    st.write(summary.metadata.document_status)
                    
                    st.markdown("**Document Author**")
                    st.write(summary.metadata.document_author)

                ## Person
                st.markdown("### Person")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Person Name**")
                    st.write(summary.person.name)

                    st.markdown("**Person Role**")
                    st.write(summary.person.role)

                with col2:
                    st.markdown("**Person Email**")
                    st.write(summary.person.email)

                ## Problem
                st.markdown("### Problem")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Vision & Opportunity**")
                    st.write(summary.problem.vision_opportunity)

                with col2:
                    st.markdown("**Target Use Case**")
                    st.write(summary.problem.target_use_case)

                ## Solution
                st.markdown("### Solution")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Goals**")
                    st.write(summary.solution.goals)

                    st.markdown("**Conceptual Model**")
                    st.write(summary.solution.conceptual_model)

                with col2:
                    st.markdown("**Requirements**")
                    st.write(summary.solution.requirements)

                with st.expander("Original Document"):
                    st.text(text[:500] + "...")

                with st.expander("JSON Output"):
                    st.json(summary.model_dump())
                
                # Download JSON
                json_str = summary.model_dump_json(indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{uploaded_file.name.split('.')[0]}_metadata.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")
                st.write("Please make sure you've uploaded a valid PDF file and that your API keys are correctly set.")
else:
    st.info("Please upload a PDF file to begin.")

st.markdown("---")