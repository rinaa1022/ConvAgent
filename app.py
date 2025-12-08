"""
Resume Parser & Knowledge Graph Builder
Main application for parsing resumes and building a Neo4j knowledge graph
"""

import streamlit as st
import os
import sys
import re
from typing import List, Dict, Any, Optional
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
from job_cache import get_jobs, JobSourceError, get_cache_age_hours

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
if 'pending_second_batch' not in st.session_state:
    st.session_state.pending_second_batch = None
# Track processed files to avoid duplicates
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'uploader_position' not in st.session_state:
    st.session_state.uploader_position = 'bottom'
if 'latest_job_intent' not in st.session_state:
    st.session_state.latest_job_intent = {"is_job_request": False}


def move_uploader_to_top():
    """Ensure the uploader docks below the JobTalk button after first use."""
    st.session_state.uploader_position = 'top'

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


def detect_job_application_intent(prompt: str) -> Dict[str, Any]:
    """Detect if the user is asking for job applications and normalize the role."""
    if not prompt:
        return {"is_job_request": False}

    lowered = prompt.lower()
    intent_keywords = ["apply", "application", "looking for", "fetch", "find me", "show me", "search for", "refresh", "update", "return", "list", "give me", "jobs", "roles", "positions", "openings", "opportunities"]
    if not any(keyword in lowered for keyword in intent_keywords):
        return {"is_job_request": False}

    role_mappings = {
        "data science intern": "data_science_intern",
        "machine learning intern": "data_science_intern",
        "ml intern": "data_science_intern",
        "ai intern": "data_science_intern",
        "software engineer intern": "software_engineer_intern",
        "software engineering intern": "software_engineer_intern",
        "swe intern": "software_engineer_intern",
        "intern": "software_engineer_intern",
        "internship": "software_engineer_intern",
        "data science": "data_science_full_time",
        "machine learning": "data_science_full_time",
        "ml engineer": "data_science_full_time",
        "ai": "data_science_full_time",
        "software engineer": "software_engineer_full_time",
        "software engineering": "software_engineer_full_time",
        "backend": "software_engineer_full_time",
        "frontend": "software_engineer_full_time",
        "full-time": "software_engineer_full_time",
        "data analyst": "data_science_full_time",
        "data engineer": "data_science_full_time",
        "product manager": "product_management_full_time",
        "product management": "product_management_full_time",
        "product intern": "product_management_intern",
        "product management intern": "product_management_intern",
        "pm intern": "product_management_intern",
        "quant": "quant_full_time",
        "quantitative": "quant_full_time",
        "trading": "quant_full_time",
        "hardware": "hardware_full_time",
        "electrical engineer": "hardware_full_time",
        "embedded": "hardware_full_time",
    }

    matched_role = None
    raw_role_phrase = None
    for phrase, normalized in role_mappings.items():
        if phrase in lowered:
            matched_role = normalized
            raw_role_phrase = phrase
            break

    # Detect if user asked for a specific number of roles
    max_results = 5
    count_match = re.search(r"(?:show|list|give|find)\s+(\d{1,2})\s+(?:roles|jobs|positions)", lowered)
    if count_match:
        try:
            requested = int(count_match.group(1))
            if 1 <= requested <= 20:
                max_results = requested
        except ValueError:
            pass

    refresh_requested = False
    if "refresh" in lowered and any(term in lowered for term in ["job", "role", "listing", "posting"]):
        refresh_requested = True
    if "update" in lowered and any(term in lowered for term in ["job", "role", "listing", "posting"]):
        refresh_requested = True

    return {
        "is_job_request": True,
        "normalized_role": matched_role,
        "raw_role_phrase": raw_role_phrase,
        "original_prompt": prompt,
        "jobs": [],
        "error": None,
        "max_results": max_results,
        "refresh_requested": refresh_requested,
        "cache_age_hours": None,
        "is_stale": False,
        "refresh_performed": False,
    }

def extract_location_from_prompt(prompt: str) -> Optional[str]:
    """
    Very simple heuristic for locations like:
    - 'in Seattle'
    - 'in California'
    - 'in New York'
    Returns a cleaned location string or None.
    """
    lowered = prompt.lower()
    m = re.search(r"\bin\s+([a-z\s,]+)", lowered)
    if not m:
        return None

    loc = m.group(1)

    # Strip trailing generic words
    stop_words = ["jobs", "job", "roles", "positions", "internships", "internship"]
    for w in stop_words:
        loc = loc.replace(w, "")

    loc = loc.strip(" ,")
    return loc or None


ROLE_SOURCE_MAP = {
    "software_engineer_intern": ("internship", "software_engineering"),
    "software_engineer_full_time": ("full_time", "software_engineering"),
    "data_science_intern": ("internship", "data_science"),
    "data_science_full_time": ("full_time", "data_science"),
    "product_management_intern": ("internship", "product_management"),
    "product_management_full_time": ("full_time", "product_management"),
    "quant_full_time": ("full_time", "quant"),
    "hardware_full_time": ("full_time", "hardware"),
}


def format_job_suggestions(job_intent: Dict[str, Any]) -> Optional[str]:
    """Return a markdown string with job suggestions, if available."""
    if not job_intent.get("is_job_request"):
        return None

    jobs = job_intent.get("jobs") or []
    error = job_intent.get("error")
    if error:
        return f"\n\n**Latest opportunities:**\n- Unable to fetch job listings right now: {error}"

    if not jobs:
        return "\n\n**Latest opportunities:**\n- No matching roles found in the repositories right now."

    max_results = job_intent.get("max_results", 5)
    lines = ["\n\n**Latest opportunities:**"]
    for posting in jobs[: max_results]:
        link = posting.apply_url or "No direct link available"
        lines.append(f"- {posting.company} – {posting.role} ({posting.location}) – [Apply]({link})")

    cache_age_hours = job_intent.get("cache_age_hours")
    if job_intent.get("refresh_performed"):
        lines.append("_Fetched the latest listings just now._")
    elif job_intent.get("is_stale") and cache_age_hours is not None:
        age_display = f"{cache_age_hours:.1f}".rstrip("0").rstrip(".")
        lines.append(f"_Listings last updated ~{age_display}h ago. Say \"refresh jobs\" if you'd like me to update them._")

    return "\n".join(lines)

def interpret_semantic_score(score: float) -> str:
    """
    Human-readable label for the combined semantic score.

    This is based on coverage + SBERT + TF-IDF.
    """
    if score is None:
        return "Unknown alignment"
    if score >= 0.40:
        return "Strong alignment"
    elif score >= 0.30:
        return "Moderate alignment"
    elif score >= 0.20:
        return "Weak alignment"
    else:
        return "Very weak alignment"

def format_kg_job_suggestions(recommended_jobs: List[Dict[str, Any]]) -> str:
    """Format KG-based job recommendations (from Neo4j) as markdown."""
    if not recommended_jobs:
        return "\n\n**Matched opportunities (KG-based):**\n- No matching roles found in the knowledge graph right now."

    lines = ["\n\n**Matched opportunities (KG-based):**"]
    for job in recommended_jobs:
        title = job.get("title") or "Untitled role"
        company = job.get("company") or "Unknown company"
        location = job.get("location") or "Unknown location"
        url = job.get("apply_url") or ""

        overlap = job.get("matching_skill_count", 0)
        total = job.get("total_skill_required", 0)
        coverage = job.get("coverage", 0.0)

        matching_skills = job.get("matching_skills") or []
        missing_skills = job.get("missing_skills") or []  # NEW

        if url:
            line = f"- [{company} – {title}]({url}) ({location})"
        else:
            line = f"- {company} – {title} ({location})"

        line += f" — skill overlap: {overlap}/{total} ({coverage:.2f} coverage)"

        # show marching skills
        if matching_skills:
            line += f" — matching skills: {', '.join(matching_skills)}"

        # show missing skills
        if missing_skills:
            line += f" — missing skills: {', '.join(missing_skills)}"

        semantic_score = job.get("combined_score")
        if semantic_score is not None:
            label = interpret_semantic_score(semantic_score)
            line += f" — Semantic score: {semantic_score:.2f} ({label})"

        lines.append(line)

    return "\n".join(lines)

    
def query_llm_chat(user_question: str, resume_context: str, llm_provider: str, api_key: str) -> str:
    """Query LLM with user question and resume context"""
    # Build the prompt
    system_prompt = """
    You are a helpful career advisor assistant. You help users understand their resume, identify skill gaps, suggest job roles they're suited for, and provide career guidance based on their resume information.

    You have access to:
    - The user's resume data (parsed into a structured knowledge graph).
    - Separate job-matching logic that may append matched job opportunities
    after your answer.

    Your job is to:
    - Analyze the resume, skills, experience, and education.
    - Suggest job roles and internship roles the user is a strong fit for.
    - Explain why certain roles fit based on their background.
    - Identify skill gaps and recommend what to learn next.
    - Give concrete, encouraging career advice.

    Important:
    - DO NOT say that you lack access to real-time job data or job postings.
    - DO NOT mention browsing the internet or limitations about not seeing current jobs.
    - The application will handle fetching and displaying specific matched jobs
    (for example under 'Matched opportunities (KG-based)'). You should focus
    on interpreting the resume and giving guidance that complements those matches.
    """

    user_prompt = f"""{resume_context}

USER QUESTION: {user_question}

Please provide a helpful, conversational response based on the resume information above."""
    
    # Call appropriate LLM based on provider
    try:
        if llm_provider == "OpenAI":
            import openai
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model="gpt-4o",
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
        page_title="JobTalk.ai",
        page_icon="🫱🏼‍🫲🏼",
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
    
    .main .block-container {
        max-width: 800px;
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }
    
    body[data-uploader-position="bottom"] .main .block-container {
        padding-top: 4rem;
        padding-bottom: 12rem;
    }
    
    body[data-uploader-position="top"] .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 6rem;
    }
    
    /* File uploader base styling */
    div[data-testid="stFileUploader"] {
        z-index: 998;
        background: var(--background-color);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(250, 250, 250, 0.1);
        transition: bottom 0.3s ease, opacity 0.3s ease, transform 0.3s ease;
    }
    
    body[data-uploader-position="bottom"] div[data-testid="stFileUploader"] {
        position: fixed;
        bottom: 110px;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        margin-bottom: 10px;
    }
    
    body[data-uploader-position="top"] div[data-testid="stFileUploader"] {
        position: static;
        width: 100%;
        max-width: 800px;
        margin: 1rem auto 2rem;
        transform: none;
        bottom: auto;
        left: auto;
    }
    
    /* Adjust uploader opacity when messages exist and it's floating */
    body[data-uploader-position="bottom"] .main .block-container:has([data-testid="stChatMessage"]) ~ * div[data-testid="stFileUploader"],
    body[data-uploader-position="bottom"] .main .block-container:has([data-testid="stChatMessage"]) + * div[data-testid="stFileUploader"] {
        opacity: 0.7;
    }
    
    /* Chat input styling - ensure it doesn't overlap content */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 800px;
        z-index: 999;
        background: var(--background-color);
        padding: 1rem;
        border-top: 1px solid rgba(250, 250, 250, 0.1);
    }
    
    /* Ensure chat messages have proper spacing */
    [data-testid="stChatMessage"] {
        margin-bottom: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <script>
    document.body.setAttribute('data-uploader-position', '{st.session_state.uploader_position}');
    
    // Dynamically adjust uploader position as content grows (bottom mode only)
    function adjustUploaderPosition() {{
        const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
        const uploader = document.querySelector('div[data-testid="stFileUploader"]');
        const chatInput = document.querySelector('.stChatInput');
        const uploaderPosition = document.body.getAttribute('data-uploader-position');
        
        if (!uploader || !chatInput) {{
            return;
        }}
        
        if (uploaderPosition === 'top') {{
            uploader.style.bottom = '';
            uploader.style.left = '';
            uploader.style.transform = '';
            uploader.style.position = '';
            return;
        }}
        
        if (chatMessages.length > 0) {{
            let totalHeight = 0;
            chatMessages.forEach(msg => {{
                totalHeight += msg.offsetHeight;
            }});
            
            const windowHeight = window.innerHeight;
            const chatInputHeight = chatInput.offsetHeight;
            const uploaderHeight = uploader.offsetHeight;
            const gapBetween = 25;
            
            const contentBottom = totalHeight;
            const availableSpace = windowHeight - chatInputHeight - uploaderHeight - gapBetween - 20;
            
            if (contentBottom > availableSpace) {{
                const moveUp = Math.min(contentBottom - availableSpace + 110, windowHeight * 0.4);
                uploader.style.bottom = (110 + moveUp) + 'px';
            }} else {{
                uploader.style.bottom = '110px';
            }}
        }}
    }}
    
    window.addEventListener('load', adjustUploaderPosition);
    
    const observer = new MutationObserver(function(mutations) {{
        let shouldUpdate = false;
        mutations.forEach(function(mutation) {{
            if (mutation.addedNodes.length || mutation.type === 'childList') {{
                shouldUpdate = true;
            }}
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {{
                const target = mutation.target;
                if (target.tagName === 'BUTTON' || target.closest('div[data-testid="stButton"]')) {{
                    shouldUpdate = true;
                }}
            }}
        }});
        if (shouldUpdate) {{
            setTimeout(function() {{
                adjustUploaderPosition();
            }}, 50);
        }}
    }});
    
    setTimeout(function() {{
        const container = document.querySelector('.main .block-container');
        if (container) {{
            observer.observe(container, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['style'] }});
        }}
        observer.observe(document.body, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['style'] }});
        adjustUploaderPosition();
    }}, 500);
    
    window.addEventListener('resize', adjustUploaderPosition);
    window.addEventListener('scroll', adjustUploaderPosition);
    </script>
    """, unsafe_allow_html=True)
    
    
    
    # Setup button - centered at top, displays "JobTalk!"
    # Use columns to center the button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("JobTalk!", key="setup_button", use_container_width=True):
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
        connection_status = None
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Test Connection", use_container_width=True):
                try:
                    test_manager = UnifiedNeo4jManager(neo4j_uri, neo4j_user, neo4j_password)
                    if test_manager.test_connection():
                        connection_status = ("success", "✅ Connection successful!")
                        test_manager.close()
                    else:
                        connection_status = ("error", "❌ Connection failed!")
                except Exception as e:
                    connection_status = ("error", f"❌ Error: {e}")
        
        with col2:
            if st.button("Save & Start", type="primary", use_container_width=True):
                if api_key and neo4j_uri and neo4j_user and neo4j_password:
                    if initialize_components(llm_provider, api_key, neo4j_uri, neo4j_user, neo4j_password):
                        st.success("✅ Ready!")
                        st.rerun()
                else:
                    st.error("Please fill all fields")
        
        if connection_status:
            status_type, status_message = connection_status
            if status_type == "success":
                st.success(status_message)
            elif status_type == "error":
                st.error(status_message)
        
        st.markdown("---")
    
    # Check if setup is complete
    if not st.session_state.setup_complete:
        return
    
    # File uploader - defined early but positioned at bottom via CSS
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        key="resume_upload",
        on_change=move_uploader_to_top,
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
        # Always store user message first
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show user bubble
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant bubble
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    # 1) No resume yet → just show instructions, skip everything else
                    if not st.session_state.current_resume_id:
                        response = """I can help you parse and explore resumes.

                        **To get started:**
                        1. Upload a resume file using the file uploader below
                        2. Once parsed, I can answer questions about that resume and match it to jobs

                        **You can ask questions like:**
                        - "What job roles am I a strong fit for?"
                        - "Find software engineering internships that match my skills."
                        - "What skills am I lacking for a data scientist role?"
                        - "What career path would suit me?"

                        Upload a resume to begin!"""
                        final_response = response

                    else:
                        # We have a resume → get resume context
                        resume_context = get_resume_context(st.session_state.current_resume_id)

                        llm_provider = st.session_state.config_llm_provider
                        api_key = st.session_state.config_api_key

                        # Call LLM for the main conversational answer
                        response = query_llm_chat(prompt, resume_context, llm_provider, api_key)

                        # Detect job intent and fetch jobs
                        job_intent = detect_job_application_intent(prompt)
                        st.session_state.latest_job_intent = job_intent

                        recommended_jobs_markdown = ""

                        if job_intent.get("is_job_request"):
                            normalized_role = job_intent.get("normalized_role")
                            source_info = (
                                ROLE_SOURCE_MAP.get(normalized_role)
                                if normalized_role
                                else None
                            )

                            # Only hit GitHub / Neo4j if we have a mapped role
                            if source_info:
                                source_type, category = source_info
                                try:
                                    # --- Fetch GitHub jobs into cache + Neo4j ---
                                    cache_age = get_cache_age_hours(source_type, category)
                                    job_intent["cache_age_hours"] = cache_age
                                    stale_threshold = 1.0
                                    job_intent["is_stale"] = (cache_age is not None and cache_age >= stale_threshold)

                                    force_refresh = job_intent.get("refresh_requested", False)
                                    job_intent["jobs"] = get_jobs(source_type, category, force_refresh=force_refresh, page=1, page_size=50)
                                    job_intent["error"] = None

                                    if force_refresh:
                                        job_intent["refresh_performed"] = True
                                        job_intent["cache_age_hours"] = (get_cache_age_hours(source_type, category) or 0.0)
                                        job_intent["is_stale"] = False
                                    else:
                                        job_intent["refresh_performed"] = False
                                        if cache_age is None:
                                            job_intent["cache_age_hours"] = (get_cache_age_hours(source_type, category) or 0.0)
                                except JobSourceError as exc:
                                    job_intent["jobs"] = []
                                    job_intent["error"] = f"Job source error: {exc}"
                                except Exception as exc:
                                    job_intent["jobs"] = []
                                    job_intent["error"] = f"Unexpected error while fetching jobs: {exc}"

                                # --- KG-based recommendations from Neo4j ---
                                try:
                                    # Extract location from the user prompt, e.g. "in California", "in Seattle"
                                    user_location = extract_location_from_prompt(prompt)

                                    max_results = job_intent.get("max_results", 5)

                                    result = st.session_state.unified_manager.recommend_jobs_for_person(
                                        person_id=st.session_state.current_resume_id,
                                        limit=max_results,
                                        source=source_type,
                                        category=category,
                                        location=user_location,  # string or None
                                    )

                                    recommended = result.get("selected_jobs", [])
                                    meta = result.get("meta", {}) or {}

                                    recommended_jobs_markdown = format_kg_job_suggestions(recommended)

                                    # only show skill-gap suggestion if *no* jobs were returned
                                    top_skills = meta.get("top_skills_from_jobs") or []
                                    improvement_skills = meta.get("improvement_skills") or []

                                    if not recommended and (improvement_skills or top_skills):
                                        skills_to_show = improvement_skills or top_skills
                                        recommended_jobs_markdown += (
                                            "\n\n_There were no strong matches in the given roles._ "
                                            "Based on those jobs, you might want to strengthen: "
                                            + ", ".join(f"**{s}**" for s in skills_to_show)
                                        )


                                except Exception as e:
                                    st.error(f"KG-based matching failed: {e}")
                                    recommended_jobs_markdown = "\nMatched opportunities (KG-based):\n\n- Unable to fetch job matches."


                            # Clear intent so it’s not reused next time
                            st.session_state.latest_job_intent = {
                                "is_job_request": False
                            }

                        # Combine LLM answer + job list
                        if recommended_jobs_markdown:
                            final_response = f"{response}{recommended_jobs_markdown}"
                        else:
                            final_response = response

                    # Render and store assistant message
                    st.markdown(final_response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_response}
                    )

            except Exception as outer_exc:
                # Catch ANY error so Streamlit doesn’t silently die
                import traceback

                traceback.print_exc()
                st.error(
                    f"Something went wrong while handling your message: {outer_exc}"
                )



if __name__ == "__main__":
    main()
