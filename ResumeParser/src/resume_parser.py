import os
import json
import re
import requests
from typing import Optional, Dict, Any
import PyPDF2
from docx import Document
try:
    from .resume_schema import ResumeData
except ImportError:
    from resume_schema import ResumeData
import openai
import google.generativeai as genai

class ResumeParser:
    def __init__(self, llm_provider: str, api_key: str):
        self.llm_provider = llm_provider
        self.api_key = api_key
        self._setup_llm()
    
    def _setup_llm(self):
        """Initialize the selected LLM provider"""
        if self.llm_provider == "OpenAI":
            openai.api_key = self.api_key
        elif self.llm_provider == "Anthropic":
            # Anthropic uses direct API calls, no setup needed
            pass
        elif self.llm_provider == "Google":
            genai.configure(api_key=self.api_key)
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension in ['.doc', '.docx']:
            return self._extract_from_docx(file_path)
        elif file_extension == '.txt':
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def parse_resume_with_llm(self, raw_text: str) -> ResumeData:
        """Parse resume text using the selected LLM"""
        prompt = self._create_parsing_prompt(raw_text)
        
        if self.llm_provider == "OpenAI":
            response = self._call_openai(prompt)
        elif self.llm_provider == "Anthropic":
            response = self._call_anthropic(prompt)
        elif self.llm_provider == "Google":
            response = self._call_google(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        
        return self._parse_llm_response(response)
    
    def _create_parsing_prompt(self, raw_text: str) -> str:
        """Create a detailed prompt for resume parsing"""
        return f"""
You are an expert resume parser. Parse the following resume text and extract structured information.

Resume Text:
{raw_text}

Please extract the following information and return it as a JSON object following this exact schema:

{{
    "personal_info": {{
        "name": "Full Name",
        "email": "email@example.com",
        "phone": "phone number",
        "address": "full address",
        "linkedin": "LinkedIn profile URL if available",
        "github": "GitHub profile URL if available"
    }},
    "summary": "Professional summary or objective statement",
    "education": [
        {{
            "institute": "University/College Name",
            "degree": "Degree Type (Bachelor's, Master's, etc.)",
            "major": ["Major Field 1", "Major Field 2"],
            "dates": {{
                "from_date": "YYYY-MM",
                "to_date": "YYYY-MM or Present"
            }},
            "courses": ["Course 1", "Course 2"],
            "gpa": "GPA if mentioned"
        }}
    ],
    "experience": [
        {{
            "position": "Job Title",
            "company": "Company Name",
            "dates": {{
                "from_date": "YYYY-MM",
                "to_date": "YYYY-MM or Present"
            }},
            "description": "Detailed job description and responsibilities",
            "skills_used": ["Skill 1", "Skill 2"],
            "location": "Work location if mentioned"
        }}
    ],
    "skills": [
        {{
            "name": "Skill Name",
            "category": "Technical/Soft/Language",
            "proficiency": "Beginner/Intermediate/Advanced if mentioned"
        }}
    ],
    "projects": [
        {{
            "name": "Project Name",
            "description": "Project description",
            "technologies": ["Tech 1", "Tech 2"],
            "dates": {{
                "from_date": "YYYY-MM",
                "to_date": "YYYY-MM"
            }},
            "url": "Project URL if available"
        }}
    ],
    "certifications": [
        {{
            "name": "Certification Name",
            "issuer": "Issuing Organization",
            "date": "YYYY-MM",
            "expiry": "YYYY-MM if applicable"
        }}
    ],
    "languages": ["Language 1", "Language 2"],
    "achievements": ["Achievement 1", "Achievement 2"]
}}

Important instructions:
1. Extract ALL information from the resume text
2. If a field is not present, use an empty array [] - NEVER use null values in arrays
3. For dates, use YYYY-MM format or "Present" for current positions
4. Be thorough in extracting skills, especially technical skills
5. Include all work experiences, even if brief
6. Extract all educational qualifications
7. For arrays (languages, achievements, skills, etc.), only include actual values - skip empty or missing items
8. Return ONLY the JSON object, no additional text or formatting
"""
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            response = openai.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=8000  # Increased to prevent truncation
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API using direct HTTP requests"""
        # Validate inputs
        if not self.api_key:
            raise Exception("Anthropic API key is not set")
        
        if not prompt or not prompt.strip():
            raise Exception("Prompt cannot be empty")
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        # Truncate prompt if it's extremely long (approximate token limit consideration)
        # Claude-3-Haiku has context window of 200K tokens, but we need to leave room for response
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = 150000  # Leave room for response tokens
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[Note: Resume text truncated due to length]"

        data = {
            "model": "claude-3-haiku-20240307",
            "max_tokens": 4096,  # Claude-3-Haiku max output tokens limit
            "temperature": 0,
            "system": (
                "You are a resume parser. Return valid JSON only—no prose. "
                "If information is missing, use null. Follow the exact schema provided."
            ),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        }

        try:
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data)
            
            # Check for errors and provide detailed error message
            if resp.status_code != 200:
                try:
                    error_data = resp.json()
                    error_msg = error_data.get("error", {})
                    
                    # Handle different error formats
                    if isinstance(error_msg, dict):
                        error_detail = f"Status {resp.status_code}: {error_msg.get('message', 'Unknown error')}"
                        if "type" in error_msg:
                            error_detail += f" (Type: {error_msg['type']})"
                    else:
                        error_detail = f"Status {resp.status_code}: {str(error_msg)}"
                    
                    # Include request details for debugging
                    error_detail += f"\nModel: {data['model']}, Max tokens: {data['max_tokens']}"
                    
                    raise Exception(f"Anthropic API error: {error_detail}")
                except (json.JSONDecodeError, AttributeError):
                    # If we can't parse error, return raw response
                    error_text = resp.text[:500]  # First 500 chars
                    raise Exception(f"Anthropic API request failed: {resp.status_code} {resp.reason}\nResponse: {error_text}")
            
            response_data = resp.json()
            text = response_data.get("content", [{"text": ""}])[0].get("text", "")
            
            # Attempt to isolate a JSON object/array
            json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
            if not json_match:
                return text.strip()

            json_str = json_match.group(0)
            try:
                # Try to parse as JSON to validate
                json.loads(json_str)
                return json_str
            except json.JSONDecodeError:
                # Clean up the JSON string
                cleaned = json_str.replace("\n", " ").replace("\t", " ")
                cleaned = re.sub(r",\s*}", "}", cleaned)
                cleaned = re.sub(r",\s*]", "]", cleaned)
                try:
                    json.loads(cleaned)
                    return cleaned
                except Exception:
                    return text.strip()
                    
        except requests.exceptions.RequestException as e:
            raise Exception(f"Anthropic API request failed: {str(e)}")
        except Exception as e:
            if "Anthropic API" in str(e):
                raise  # Re-raise our formatted errors
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def _call_google(self, prompt: str) -> str:
        """Call Google Gemini API"""
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=8000,  # Increased to prevent truncation
                    temperature=0.1
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text that might contain markdown or other content"""
        text = text.strip()
        
        # Remove markdown code blocks
        if '```json' in text:
            # Find JSON code block
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        elif '```' in text:
            # Generic code block
            start = text.find('```') + 3
            end = text.find('```', start)
            if end != -1:
                text = text[start:end].strip()
        
        # Try to find JSON object boundaries
        # Look for first { and matching last }
        first_brace = text.find('{')
        if first_brace == -1:
            return text
        
        # Find matching closing brace
        brace_count = 0
        last_brace = first_brace
        for i, char in enumerate(text[first_brace:], start=first_brace):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    last_brace = i + 1
                    break
        
        if brace_count == 0:
            return text[first_brace:last_brace]
        else:
            # Unmatched braces - might be truncated, return what we have
            return text[first_brace:]
    
    def _fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON issues"""
        original = json_str
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)  # Run twice to catch nested cases
        
        # Fix missing commas between array items: ] [ should be ], [
        json_str = re.sub(r']\s+\[', '], [', json_str)
        
        # Fix missing commas after closing quotes followed by opening quote (field,field)
        # Pattern: "value""key" should be "value","key"
        json_str = re.sub(r'("\s*)(?=")', r'\1,', json_str)
        
        # Fix missing commas after values before new keys
        # Pattern: "value"\s+"key" should be "value", "key"
        json_str = re.sub(r'("\s*)\n\s*"', r'\1,\n        "', json_str)
        
        # Fix missing commas after closing braces/brackets before new keys
        # Pattern: }\s+"key" should be }, "key"
        json_str = re.sub(r'(}\s+)(?=")', r'\1, ', json_str)
        json_str = re.sub(r'(]\s+)(?=")', r'\1, ', json_str)
        
        # Fix missing commas after closing braces before arrays
        # Pattern: }\s+\[ should be }, [
        json_str = re.sub(r'}\s+\[', '}, [', json_str)
        
        # Fix missing commas after closing quotes before braces/brackets
        # Pattern: "\s+{ should be ", { (but not if it's "key": { which is valid)
        json_str = re.sub(r'("\s+)(?={)', r'\1, ', json_str)
        
        # Fix missing commas after numbers before keys or closing braces
        json_str = re.sub(r'(\d)\s+"', r'\1, "', json_str)
        json_str = re.sub(r'(\d)\s+}', r'\1}', json_str)  # Don't add comma before closing brace
        json_str = re.sub(r'(\d)\s+]', r'\1]', json_str)  # Don't add comma before closing bracket
        
        # Fix missing commas after true/false/null before keys or closing braces
        json_str = re.sub(r'\b(true|false|null)\s+"', r'\1, "', json_str)
        json_str = re.sub(r'\b(true|false|null)\s+}', r'\1}', json_str)
        json_str = re.sub(r'\b(true|false|null)\s+]', r'\1]', json_str)
        
        # Try to fix unterminated strings at the end (truncation)
        # Count quotes to detect if we're in the middle of a string
        lines = json_str.split('\n')
        if lines:
            last_line = lines[-1]
            quote_count = last_line.count('"') - last_line.count('\\"')
            # If odd number of quotes, we might be in a string
            if quote_count % 2 == 1 and not json_str.strip().endswith('"'):
                # Try to close the string and object
                if '"' in last_line and ':' in last_line:
                    # Likely an unfinished value
                    json_str = json_str.rstrip()
                    # Find the last unclosed quote
                    last_quote_idx = json_str.rfind('"')
                    if last_quote_idx != -1:
                        # Close the string
                        json_str = json_str[:last_quote_idx + 1] + '"'
                        # Try to close any open structures
                        open_braces = json_str.count('{') - json_str.count('}')
                        open_brackets = json_str.count('[') - json_str.count(']')
                        # Close brackets first, then braces
                        json_str += ']' * open_brackets
                        json_str += '}' * open_braces
        
        return json_str
    
    def _fix_delimiter_error(self, json_str: str, error_pos: int) -> str:
        """Try to fix delimiter errors at a specific position"""
        if error_pos >= len(json_str):
            return json_str
        
        # Get context around error position
        context_start = max(0, error_pos - 50)
        context_end = min(len(json_str), error_pos + 50)
        context = json_str[context_start:context_end]
        
        # Try to insert comma at various positions near the error
        fixes_to_try = []
        
        # Look for patterns that suggest missing comma
        # Pattern 1: "value" "key" (missing comma between string values)
        if '"' in context:
            # Find positions where we might need a comma
            for i in range(max(0, error_pos - 10), min(len(json_str), error_pos + 10)):
                if i < len(json_str) - 1:
                    # Check for: quote followed by whitespace and another quote
                    if json_str[i] == '"' and json_str[i+1:i+2].isspace() and i+2 < len(json_str) and json_str[i+2] == '"':
                        # Check if there's already a comma
                        if json_str[i+1:i+2] != ',':
                            # Insert comma
                            fixes_to_try.append((i+1, json_str[:i+1] + ',' + json_str[i+1:]))
        
        # Pattern 2: } or ] followed by " (missing comma before new key)
        for i in range(max(0, error_pos - 10), min(len(json_str), error_pos + 10)):
            if i < len(json_str) - 1:
                if json_str[i] in '}]' and json_str[i+1:i+2].isspace() and i+2 < len(json_str) and json_str[i+2] == '"':
                    # Skip if it's part of "key": { or "key": [
                    if i > 0 and json_str[i-1] != ':':
                        if json_str[i+1:i+2] != ',':
                            fixes_to_try.append((i+1, json_str[:i+1] + ', ' + json_str[i+1:]))
        
        # Pattern 3: Number or true/false/null followed by " (missing comma)
        for i in range(max(0, error_pos - 10), min(len(json_str), error_pos + 10)):
            if i < len(json_str) - 1:
                # Check if it's a number, true, false, or null followed by quote
                if json_str[i].isdigit() and json_str[i+1:i+2].isspace() and i+2 < len(json_str) and json_str[i+2] == '"':
                    if json_str[i+1:i+2] != ',':
                        fixes_to_try.append((i+1, json_str[:i+1] + ', ' + json_str[i+1:]))
                elif i >= 4 and json_str[i-4:i+1] in ['true', 'false', 'null']:
                    if i+1 < len(json_str) and json_str[i+1].isspace() and i+2 < len(json_str) and json_str[i+2] == '"':
                        if json_str[i+1:i+2] != ',':
                            fixes_to_try.append((i+1, json_str[:i+1] + ', ' + json_str[i+1:]))
        
        # Try each fix and see if it parses
        for pos, fixed in fixes_to_try:
            try:
                json.loads(fixed)
                return fixed  # If it parses, return the fixed version
            except:
                continue
        
        return json_str  # If no fix worked, return original
    
    def _parse_llm_response(self, response: str) -> ResumeData:
        """Parse LLM response and create ResumeData object"""
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_text(response)
            
            # Try to parse as-is first
            try:
                data = json.loads(json_str)
                return ResumeData(**data)
            except json.JSONDecodeError:
                # Try fixing common issues
                fixed_json = self._fix_json_string(json_str)
                try:
                    data = json.loads(fixed_json)
                    return ResumeData(**data)
                except json.JSONDecodeError as e:
                    # If still failing, try to fix delimiter errors at specific position
                    error_msg = str(e)
                    
                    # Try to find error position from error message
                    match = re.search(r'char (\d+)', error_msg)
                    if match and "Expecting" in error_msg and "delimiter" in error_msg:
                        pos = int(match.group(1))
                        # Try to fix delimiter error at this position
                        delimiter_fixed = self._fix_delimiter_error(json_str, pos)
                        if delimiter_fixed != json_str:
                            try:
                                data = json.loads(delimiter_fixed)
                                return ResumeData(**data)
                            except:
                                pass
                        
                        # Also try fixing at position in the already-fixed version
                        delimiter_fixed2 = self._fix_delimiter_error(fixed_json, pos)
                        if delimiter_fixed2 != fixed_json:
                            try:
                                data = json.loads(delimiter_fixed2)
                                return ResumeData(**data)
                            except:
                                pass
                    
                    # If still failing, try truncation for unterminated strings
                    if "Unterminated string" in error_msg or ("Expecting" in error_msg and match):
                        # Try to truncate at the error position
                        if match:
                            pos = int(match.group(1))
                            truncated = json_str[:pos]
                            # Try to close any open structures
                            truncated = truncated.rstrip().rstrip(',').rstrip()
                            open_braces = truncated.count('{') - truncated.count('}')
                            open_brackets = truncated.count('[') - truncated.count(']')
                            # Close brackets first, then braces
                            truncated += ']' * open_brackets
                            truncated += '}' * open_braces
                            try:
                                data = json.loads(truncated)
                                return ResumeData(**data)
                            except:
                                pass
                    
                    raise Exception(f"Failed to parse JSON response: {error_msg}\n\nResponse preview (first 500 chars):\n{response[:500]}")
            
        except Exception as e:
            if "Failed to parse JSON" in str(e):
                raise
            raise Exception(f"Failed to create ResumeData object: {str(e)}")
