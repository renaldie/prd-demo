import os
import tempfile
import json
import streamlit as st
import re
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from markitdown import MarkItDown

# Set page title and configure the page
st.set_page_config(page_title="Smart PRD", layout="centered")
st.title("Smart-PRD :computer:")
st.write("AI-Powered Requirement Management Engineer")
st.markdown("---")

# Load environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or st.secrets.get("GITHUB_TOKEN")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY") or st.secrets.get("OPEN_AI_API_KEY")

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
    document_title: Optional[str] = Field(default=None, description="The name of the document")
    document_date: Optional[str] = Field(default=None, description="The date of the document")
    document_status: Optional[str] = Field(default=None, description="The status of the document")
    document_author: Optional[str] = Field(default=None, description="The author of the document")

class Person(BaseModel):
    name: Optional[str] = Field(default=None, description="The name of the person beside the author")
    role: Optional[str] = Field(default=None, description="The role of the person beside the author")
    email: Optional[str] = Field(default=None, description="The email of the person beside the author")

class Problem(BaseModel):
    vision_opportunity: Optional[str] = Field(default=None, description="The Vision & Opportunity of the document")
    target_use_case: Optional[str] = Field(default=None, description="The Target Use Case of the document")

class Solution(BaseModel):
    goals: Optional[str] = Field(default=None, description="The goals of the solution")
    conceptual_model: Optional[str] = Field(default=None, description="The conceptual model of the solution")
    requirements: Optional[str] = Field(default=None, description="The requirements of the solution")

class Summary(BaseModel):
    metadata: Metadata = Field(description="The metadata of the document")
    person: Person = Field(description="The person in the document")
    problem: Problem = Field(description="The problem in the document")
    solution: Solution = Field(description="The solution in the document")

summary_parser = PydanticOutputParser(pydantic_object=Summary)

def extract_prd(prd, max_retries=3):
    """Extract PRD info with simple retry mechanism"""
    
    prd_template = """"
    You are an expert in analyzing Product Requirements Documents (PRDs).
    Your task is to extract software requirements from the provided PRD text.
    
    PRD text: {prd}
    
    For each PRD text, find the following information:
    - metadata: object with document_title, document_date, document_status, document_author
    - person: object with name, role, email of key stakeholders
    - problem: object with vision_opportunity and target_use_case
    - solution: object with goals, conceptual_model, and requirements

    IMPORTANT: Format all above-mentioned objects as string or list of strings only.
    If a section doesn't contain anything, output null for that section.
    """

    prd_prompt_template = PromptTemplate(
        input_variables=["prd"],
        template=prd_template
        )

    LLM = get_llm()
    chain = prd_prompt_template | LLM.with_structured_output(schema=Summary, include_raw=True)
    
    # Simple retry logic
    for attempt in range(max_retries):
        try:
            output = chain.invoke(input={"prd": prd})
            return output  # Return immediately on success
        except Exception as e:
            if attempt == max_retries - 1:  # If this was the last attempt
                st.error(f"All attempts failed. Last error: {str(e)}")
                return None  # Return None after all retries fail
    
    # Should never reach here, but just in case
    return None

def clean_text(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    json_str = match.group(0)
    json_obj = json.loads(json_str)
    return Summary.model_validate(json_obj)

def convert_to_md(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    md = MarkItDown()
    result = md.convert(tmp_file_path)
    return result.text_content

def display_result(original_text, summary, file_object):
    ## Metadata
    st.markdown("### Metadata")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Document Title**")
        st.write(summary['parsed'].metadata.document_title)
        
        st.markdown("**Document Date**")
        st.write(summary['parsed'].metadata.document_date)
    
    with col2:
        st.markdown("**Document Status**")
        st.write(summary['parsed'].metadata.document_status)
        
        st.markdown("**Document Author**")
        st.write(summary['parsed'].metadata.document_author)

    ## Person
    st.markdown("### Person")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Person Name**")
        st.write(summary['parsed'].person.name)

        st.markdown("**Person Role**")
        st.write(summary['parsed'].person.role)

    with col2:
        st.markdown("**Person Email**")
        st.write(summary['parsed'].person.email)

    ## Problem
    st.markdown("### Problem")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Vision & Opportunity**")
        st.write(summary['parsed'].problem.vision_opportunity)

    with col2:
        st.markdown("**Target Use Case**")
        st.write(summary['parsed'].problem.target_use_case)

    ## Solution
    st.markdown("### Solution")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Goals**")
        st.write(summary['parsed'].solution.goals)

        st.markdown("**Conceptual Model**")
        st.write(summary['parsed'].solution.conceptual_model)

    with col2:
        st.markdown("**Requirements**")
        st.write(summary['parsed'].solution.requirements)

    with st.expander("Original Document"):
        st.text(original_text[:500] + "...")

    with st.expander("JSON Output"):
        st.json(summary['parsed'].model_dump())
    
    # Download JSON
    json_str = summary['parsed'].model_dump_json(indent=2)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"{file_object.name.split('.')[0]}_metadata.json",
        mime="application/json"
    )

def footer_credit():
    cols = st.columns([2, 1, 2])
    with cols[1]:
        st.markdown(
            """
            <div style="text-align: center">
                © 2025 Smart PRD<br>
                Made by <a href="https://renaldi-ega.notion.site" target="_blank">Ren</a> in Hsinchu with ❤️<br>
            </div>
            """, 
            unsafe_allow_html=True
        )

def select_choose_or_upload():
    # Create a two-column layout for the buttons
    left, right = st.columns(2)

    # Create the two buttons
    if left.button("Example PRDs", type="secondary", use_container_width=True, icon="📄"):
        st.session_state.file_option = "Use example files"
        
    if right.button("Upload", type="secondary", use_container_width=True, icon="📤"):
        st.session_state.file_option = "Upload my own file"

    return st.session_state.file_option

def select_choose():
    # Initialize variables with default values
    text = None
    summary = None
    temp_file = None
    selected_example = None
    
    example_files = {
        "Homepage Change PRD": "PRD_Homepage_Change.pdf",
        "Baby Tracker App PRD": "PRD_Simple_Baby_Tracker_App.pdf"
    }
    
    selected_example = st.selectbox(label="Select an example PRD",
                                    options=list(example_files.keys()), 
                                    index=None,
                                    placeholder="Select an example PRD",
                                    label_visibility='hidden')
    
    if selected_example and st.button("Analyze PRD", type="primary", icon="🔍"):
        with st.spinner(f"Analyzing {selected_example}..."):
            example_filename = example_files[selected_example]
            file_path = os.path.join(os.path.dirname(__file__), example_filename)
            
            # Read the file content
            with open(file_path, "rb") as file:
                file_content = file.read()
            
            # Create a temporary file object similar to what st.file_uploader would provide
            class TempFile:
                def __init__(self, content, name):
                    self.content = content
                    self.name = name
                
                def getvalue(self):
                    return self.content
            
            temp_file = TempFile(file_content, example_filename)
            
            text = convert_to_md(temp_file)
            summary = extract_prd(text)

    return text, summary, temp_file, selected_example

def select_upload(uploaded_file):
    with st.spinner("Processing the PDF..."):
        text = convert_to_md(uploaded_file)
        summary = extract_prd(text)
    return text, summary, uploaded_file

# File uploader
# Replace the current pills component with these buttons
st.markdown("## 1 | Choose Example PRDs or Upload:")

# Initialize session state if needed
if "file_option" not in st.session_state:
    st.session_state.file_option = "Use example files"  # Set default here

select_choose_or_upload()

if st.session_state.file_option == "Use example files":
    text, summary, temp_file, selected_example = select_choose()
    # Only show results if analysis has been performed
    if summary is not None and text is not None and temp_file is not None:
        st.markdown("---")
        st.markdown("## 2 | PRD Details")
        display_result(original_text=text, summary=summary, file_object=temp_file)       
else:
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility='hidden')
    if uploaded_file is not None:
        text, summary, uploaded_file = select_upload(uploaded_file)
        # Only show results if we have valid data
        if summary is not None:
            st.markdown("---")
            st.markdown("## 2 | Extracted Information")
            display_result(original_text=text, summary=summary, file_object=uploaded_file)

st.markdown("---")
footer_credit()