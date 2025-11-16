"""
Resume Parser & Knowledge Graph Builder
Main application for parsing resumes and building a Neo4j knowledge graph
"""

import streamlit as st
import os
import sys
from typing import List, Dict, Any
import json
from datetime import datetime
import tempfile
import uuid

# Add modules to path
sys.path.append('ResumeParser/src')
sys.path.append('.')

from unified_neo4j_manager import UnifiedNeo4jManager
from unified_schema import normalize_skill_name
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Import resume parser components
from ResumeParser.src.resume_parser import ResumeParser

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False
if 'unified_manager' not in st.session_state:
    st.session_state.unified_manager = None
if 'resume_parser' not in st.session_state:
    st.session_state.resume_parser = None
if 'show_setup' not in st.session_state:
    st.session_state.show_setup = False
# Store configuration values
if 'config_llm_provider' not in st.session_state:
    st.session_state.config_llm_provider = "OpenAI"
if 'config_api_key' not in st.session_state:
    st.session_state.config_api_key = ""
if 'config_neo4j_uri' not in st.session_state:
    st.session_state.config_neo4j_uri = NEO4J_URI
if 'config_neo4j_user' not in st.session_state:
    st.session_state.config_neo4j_user = NEO4J_USER
if 'config_neo4j_password' not in st.session_state:
    st.session_state.config_neo4j_password = NEO4J_PASSWORD
# Track current resume
if 'current_resume_id' not in st.session_state:
    st.session_state.current_resume_id = None
if 'current_resume_name' not in st.session_state:
    st.session_state.current_resume_name = None
# Track processed files to avoid duplicates
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

def initialize_components(llm_provider: str, api_key: str, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
    """Initialize all components with configuration"""
    try:
        unified_manager = UnifiedNeo4jManager(neo4j_uri, neo4j_user, neo4j_password)
        resume_parser = ResumeParser(llm_provider, api_key)
        
        st.session_state.unified_manager = unified_manager
        st.session_state.resume_parser = resume_parser
        st.session_state.setup_complete = True
        st.session_state.show_setup = False
        
        return True
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return False
    
def parse_and_save_resume(uploaded_file):
    """Parse resume and save to unified database"""
    if not st.session_state.setup_complete:
        return {'success': False, 'error': 'Please complete setup first'}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Verify parser is initialized
        if not st.session_state.resume_parser:
            return {'success': False, 'error': 'Resume parser not initialized. Please complete setup.'}
        
        raw_text = st.session_state.resume_parser.extract_text_from_file(tmp_file_path)
        parsed_data = st.session_state.resume_parser.parse_resume_with_llm(raw_text)
        
        resume_dict = parsed_data.model_dump()
        resume_dict['id'] = str(uuid.uuid4())
        resume_dict['name'] = parsed_data.personal_info.get('name', 'Unknown')
        resume_dict['parsed_at'] = datetime.now().isoformat()
        resume_dict['original_filename'] = uploaded_file.name
        
        person_id = st.session_state.unified_manager.create_person(resume_dict)
        save_resume_to_backend(resume_dict)
        
        # Set as current resume
        st.session_state.current_resume_id = person_id
        st.session_state.current_resume_name = resume_dict['name']
        
        return {
            'success': True,
            'person_id': person_id,
            'name': resume_dict['name'],
            'data': resume_dict
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    
def save_resume_to_backend(resume_data):
    """Save parsed resume data to backend JSON file"""
    resumes_dir = "parsed_resumes"
    if not os.path.exists(resumes_dir):
        os.makedirs(resumes_dir)
    
    filename = f"{resume_data['id']}.json"
    filepath = os.path.join(resumes_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(resume_data, f, indent=2, ensure_ascii=False)

def get_resume_context(person_id: str) -> str:
    """Get formatted resume context from knowledge graph"""
    try:
        person = st.session_state.unified_manager.get_person_by_id(person_id)
        skills = st.session_state.unified_manager.get_person_skills(person_id)
        experiences = st.session_state.unified_manager.get_person_experiences(person_id)
        
        if not person:
            return "No resume data available."
        
        context = f"""RESUME INFORMATION FOR {person.get('name', 'Unknown')}:

Personal Information:
- Name: {person.get('name', 'N/A')}
- Email: {person.get('email', 'N/A')}
- Phone: {person.get('phone', 'N/A')}
- LinkedIn: {person.get('linkedin', 'N/A')}
- GitHub: {person.get('github', 'N/A')}
- Summary: {person.get('summary', 'N/A')}

Skills:"""
        
        for skill in skills:
            proficiency = f" ({skill.get('proficiency')})" if skill.get('proficiency') else ""
            context += f"\n- {skill['skill']} ({skill.get('category', 'N/A')}){proficiency}"
        
        if experiences:
            context += "\n\nWork Experience:"
            for exp in experiences:
                company = exp.get('company', 'N/A')
                position = exp.get('position', 'N/A')
                start_date = exp.get('start_date', 'N/A')
                end_date = exp.get('end_date', 'Present') if exp.get('end_date') else 'Present'
                location = exp.get('location', '')
                description = exp.get('description', '')
                
                context += f"\n- Position: {position} at {company}"
                context += f"\n  Dates: {start_date} to {end_date}"
                if location:
                    context += f"\n  Location: {location}"
                if description:
                    context += f"\n  Description: {description[:200]}..." if len(description) > 200 else f"\n  Description: {description}"
        
        if person.get('institutes'):
            context += f"\n\nEducation:\nInstitutes: {', '.join(person['institutes'])}"
        
        return context
    except Exception as e:
        return f"Error retrieving resume data: {e}"

def query_llm_chat(user_question: str, resume_context: str, llm_provider: str, api_key: str) -> str:
    """Query LLM with user question and resume context"""
    # Build the prompt
    system_prompt = """You are a helpful career advisor assistant. You help users understand their resume, identify skill gaps, suggest job roles they're suited for, and provide career guidance based on their resume information.

You have access to the user's resume data. Use this information to:
- Answer questions about their skills, experience, and education
- Suggest job roles they're a strong fit for based on their background
- Identify skills they might be lacking for specific roles
- Provide career advice and recommendations
- Be conversational, helpful, and specific

When job data is available in the future, you'll be able to match resumes to actual job postings."""
    
    user_prompt = f"""{resume_context}

USER QUESTION: {user_question}

Please provide a helpful, conversational response based on the resume information above."""
    
    # Call appropriate LLM based on provider
    try:
        if llm_provider == "OpenAI":
            import openai
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000  # Limit response length for concise answers
            )
            return response.choices[0].message.content
        
        elif llm_provider == "Anthropic":
            import requests
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            }
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1000,  # Reduced from 2000 for more concise responses
                "temperature": 0.7,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
            }
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
            resp.raise_for_status()
            response_data = resp.json()
            return response_data.get("content", [{"text": ""}])[0].get("text", "")
        
        elif llm_provider == "Google":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-pro')
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            # Google Gemini uses generation_config for token limits
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(max_output_tokens=1000)
            )
            return response.text
        
        else:
            return f"Unsupported LLM provider: {llm_provider}"
                
    except Exception as e:
        return f"Error querying LLM: {str(e)}"

def main():
    """Main entry point"""
    st.set_page_config(
        page_title="Resume Parser",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Hide all Streamlit UI elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Remove sidebar */
    section[data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 6rem;  /* Extra padding to prevent content from being hidden behind chat input */
        max-width: 800px;
    }
    
    /* Setup button styling */
    .setup-button {
        position: fixed;
        top: 1rem;
        right: 1rem;
        z-index: 1000;
    }
    
    /* Chat input styling - ensure it doesn't overlap content */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background: var(--background-color);
        padding: 1rem;
        border-top: 1px solid rgba(250, 250, 250, 0.1);
    }
    
    /* Ensure chat messages have proper spacing */
    [data-testid="stChatMessage"] {
        margin-bottom: 1.5rem;
    }
    
    /* File uploader styling - sticky at top */
    div[data-testid="stFileUploader"] {
        position: sticky;
        top: 0;
        background: var(--background-color);
        z-index: 100;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(250, 250, 250, 0.1);
    }
    
    /* Ensure proper spacing */
    .upload-container {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Setup button - fixed position top right
    if st.button("⚙️ Setup", key="setup_button", use_container_width=False):
        st.session_state.show_setup = not st.session_state.show_setup
    
    # Setup modal
    if st.session_state.show_setup or not st.session_state.setup_complete:
        with st.container():
            st.markdown("### ⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            llm_provider = st.selectbox(
                "LLM Provider",
                ["OpenAI", "Anthropic", "Google"],
                key="setup_llm_provider",
                index=["OpenAI", "Anthropic", "Google"].index(st.session_state.config_llm_provider) if st.session_state.config_llm_provider in ["OpenAI", "Anthropic", "Google"] else 0
            )
            api_key = st.text_input(
                "API Key",
                type="password",
                key="setup_api_key",
                value=st.session_state.config_api_key
            )
        
        with col2:
            neo4j_uri = st.text_input(
                "Neo4j URI",
                key="setup_neo4j_uri",
                value=st.session_state.config_neo4j_uri
            )
            neo4j_user = st.text_input(
                "Username",
                key="setup_neo4j_user",
                value=st.session_state.config_neo4j_user
            )
            neo4j_password = st.text_input(
                "Password",
                type="password",
                key="setup_neo4j_password",
                value=st.session_state.config_neo4j_password
            )
        
        # Update session state when values change
        st.session_state.config_llm_provider = llm_provider
        st.session_state.config_api_key = api_key
        st.session_state.config_neo4j_uri = neo4j_uri
        st.session_state.config_neo4j_user = neo4j_user
        st.session_state.config_neo4j_password = neo4j_password
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Connection", use_container_width=True):
                try:
                    test_manager = UnifiedNeo4jManager(neo4j_uri, neo4j_user, neo4j_password)
                    if test_manager.test_connection():
                        st.success("✅ Connection successful!")
                        test_manager.close()
                    else:
                        st.error("❌ Connection failed!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("Save & Start", type="primary", use_container_width=True):
                if api_key and neo4j_uri and neo4j_user and neo4j_password:
                    if initialize_components(llm_provider, api_key, neo4j_uri, neo4j_user, neo4j_password):
                        st.success("✅ Ready!")
                        st.rerun()
                else:
                    st.error("Please fill all fields")
        
        st.markdown("---")
    
    # Check if setup is complete
    if not st.session_state.setup_complete:
        return
    
    # File uploader - placed at top (sticky via CSS)
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        key="resume_upload",
        label_visibility="visible"
    )
    
    # Chat interface
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if uploaded_file:
        # Create a unique identifier for this file upload session
        file_id = f"{uploaded_file.name}_{uploaded_file.size}_{uploaded_file.getvalue()[:100]}"
        
        # Check if this file has already been processed
        if file_id not in st.session_state.processed_files:
            # Mark as processed immediately to prevent duplicate processing
            st.session_state.processed_files.add(file_id)
            
            st.session_state.messages.append({
                "role": "user",
                "content": f"📄 Uploaded: {uploaded_file.name}"
            })
            
            with st.chat_message("user"):
                st.write(f"📄 Uploaded: {uploaded_file.name}")
            
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    result = parse_and_save_resume(uploaded_file)
                
                if result['success']:
                    response = "✅ **Resume parsed successfully!**"
                    st.markdown(response)
                    
                    with st.expander("View Details"):
                        st.markdown(f"**Name:** {result['name']}")  
                        st.markdown(f"**ID:** `{result['person_id']}`")
                        st.markdown("")
                        st.markdown("✅ Parsed with AI")  
                        st.markdown("✅ Saved to Neo4j")  
                        st.markdown("✅ Backed up to JSON")
                        st.markdown("---")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Personal Information**")
                            st.json(result['data'].get('personal_info', {}))
                        with col2:
                            st.markdown("**Skills**")
                            skills = result['data'].get('skills', [])
                            for skill in skills:
                                st.write(f"• {skill.get('name')} ({skill.get('category')})")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    error_msg = f"❌ Error: {result.get('error')}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            # Check if there's a current resume
            if not st.session_state.current_resume_id:
                response = """I can help you parse and explore resumes.

**To get started:**
1. Upload a resume file using the file uploader above
2. Once parsed, I can answer questions about that resume

**You can ask questions like:**
- "What job roles am I a strong fit for?"
- "What skills am I lacking for a data scientist role?"
- "What are my strengths based on my resume?"
- "What career path would suit me?"

Upload a resume to begin!"""
            else:
                # Get resume context
                resume_context = get_resume_context(st.session_state.current_resume_id)
                
                # Get LLM provider and API key from session state
                llm_provider = st.session_state.config_llm_provider
                api_key = st.session_state.config_api_key
                
                # Query LLM with resume context
                with st.spinner("Thinking..."):
                    response = query_llm_chat(prompt, resume_context, llm_provider, api_key)
            
            st.markdown(response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

if __name__ == "__main__":
    main()
