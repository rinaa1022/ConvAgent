# Resume Parser & Knowledge Graph Builder

An AI-powered platform that parses resumes and builds a comprehensive Neo4j knowledge graph for advanced querying and analysis.

## 🏗️ Architecture

```
ConvAgent/
├── ResumeParser/              # Resume analysis and parsing module
│   └── src/                  # Core resume parsing code
│       ├── resume_parser.py  # Main parsing logic with robust JSON handling
│       └── resume_schema.py  # Pydantic data models
├── unified_neo4j_manager.py  # Knowledge graph manager for Neo4j
├── unified_schema.py         # Knowledge graph schema definitions
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── setup.py                  # Package setup configuration
├── requirements.txt          # Python dependencies
├── parsed_resumes/           # JSON backup storage for parsed resumes
└── README.md                 # This file
```

## 🚀 Features

### Resume Analysis
- **Multi-format Support**: PDF, DOCX, TXT files
- **AI-powered Parsing**: Uses OpenAI, Anthropic (Claude), or Google Gemini AI
- **Robust JSON Parsing**: Automatically fixes common JSON errors including:
  - Truncated responses (unterminated strings)
  - Missing delimiters (commas, brackets)
  - Malformed JSON from LLM responses
  - Markdown code block extraction
- **Skill Extraction**: Identifies technical and soft skills with proficiency levels
- **Experience Analysis**: Parses work history, companies, and positions
- **Education Parsing**: Extracts educational background and degrees
- **Knowledge Graph**: Stores structured resume data in Neo4j with relationships

### Knowledge Graph Features
- **Person Nodes**: Individual profiles with personal information
- **Skill Normalization**: Automatic skill name normalization and aliasing
- **Organization Tracking**: Companies and educational institutes
- **Position Mapping**: Job positions with skill requirements
- **Relationship Graph**: Complex relationships between entities
- **Query Interface**: Explore and query the knowledge graph

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ConvAgent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Neo4j**
   - Install [Neo4j Desktop](https://neo4j.com/download/)
   - Create a new database
   - Note the connection details (URI, username, password)

4. **Configure environment**
   ```bash
   # Create .env file or update config.py
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

## 🎯 Usage

### Run the Main Application
```bash
streamlit run app.py
```

This will start a Streamlit web interface where you can:
- **Configure Setup**: Enter API keys and Neo4j connection details
- **Upload Resumes**: Upload and parse resumes in PDF, DOCX, or TXT formats
- **View Parsed Data**: See structured resume information with personal info, skills, experience, and education
- **Chat Interface**: Ask questions about resumes using AI (e.g., "What job roles am I a strong fit for?")
- **Knowledge Graph**: Data is automatically stored in Neo4j for relationship queries

### Programmatic Usage

```python
from ResumeParser.src.resume_parser import ResumeParser
from unified_neo4j_manager import UnifiedNeo4jManager

# Initialize parser
parser = ResumeParser("OpenAI", "your_api_key")

# Parse a resume
raw_text = parser.extract_text_from_file("resume.pdf")
resume_data = parser.parse_resume_with_llm(raw_text)

# Save to Neo4j
manager = UnifiedNeo4jManager(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j",
    neo4j_password="password"
)

# Create constraints
manager.create_constraints()

# Save resume to knowledge graph
resume_dict = resume_data.model_dump()
person_id = manager.create_person(resume_dict)
```

## 🔧 Configuration

### Environment Variables
```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# API Keys (optional - can be entered in UI)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

### LLM Provider Setup

**OpenAI**
1. Sign up at https://platform.openai.com/
2. Get your API key
3. Add to environment or enter in UI

**Anthropic (Claude)**
1. Sign up at https://console.anthropic.com/
2. Get your API key (starts with `sk-ant-`)
3. Uses Claude-3-Haiku model with 4096 max output tokens
4. Add to environment or enter in UI

**Google (Gemini)**
1. Sign up at https://makersuite.google.com/
2. Get your API key
3. Uses Gemini Pro model with 8000 max output tokens
4. Add to environment or enter in UI

## 📊 Knowledge Graph Schema

### Core Entities

- **Person**: Individual with personal information, skills, and experience
- **Organization**: Companies and educational institutes
- **Position**: Job positions and roles
- **Skill**: Technical and soft skills (normalized)
- **Location**: Geographic locations

### Key Relationships

- `Person` -[:HAS_SKILL]-> `Skill`
- `Person` -[:HAS_EXPERIENCE]-> `Organization`
- `Person` -[:HAS_EDUCATION]-> `Organization`
- `Organization` -[:HAS_POSITION]-> `Position`
- `Position` -[:REQUIRES_SKILL]-> `Skill`

### Skill Normalization

The system includes a skill ontology that normalizes skill names:
- "python" = "python3", "python 3", "py"
- "javascript" = "js", "ecmascript", "node.js"
- "react" = "reactjs", "react.js"

This ensures consistent skill matching across resumes.

## 🔍 API Reference

### Resume Parser
```python
from ResumeParser.src.resume_parser import ResumeParser

parser = ResumeParser(llm_provider="OpenAI", api_key="your_key")
raw_text = parser.extract_text_from_file("resume.pdf")
resume_data = parser.parse_resume_with_llm(raw_text)
```

### Knowledge Graph Manager
```python
from unified_neo4j_manager import UnifiedNeo4jManager

manager = UnifiedNeo4jManager(neo4j_uri, neo4j_user, neo4j_password)

# Create person from resume
person_id = manager.create_person(resume_dict)

# Query methods
people = manager.get_all_people()
person = manager.get_person_by_id(person_id)
skills = manager.get_person_skills(person_id)
skill_usage = manager.find_skill_usage()
stats = manager.get_database_stats()
```

## 📈 Analytics

The platform provides insights into:
- **Skill Usage**: Most common skills across all resumes
- **Database Statistics**: Counts of entities and relationships
- **Person Profiles**: Detailed information about each person
- **Knowledge Graph Exploration**: Interactive exploration of relationships

## ✨ Recent Improvements

### JSON Parsing Robustness
- **Automatic Error Recovery**: Intelligently fixes JSON parsing errors from LLM responses
- **Truncation Detection**: Handles cases where API responses are cut off mid-string
- **Delimiter Repair**: Automatically adds missing commas and brackets at error positions
- **Multiple Fallback Strategies**: Attempts various repair methods before failing

### Code Quality
- **Cleaned Repository**: Removed unnecessary test files and utility scripts
- **Streamlined Imports**: Simplified import structure for better maintainability
- **Improved Error Messages**: More detailed error information for easier debugging

### API Support
- **Optimized Token Limits**: Proper max_tokens settings for each LLM provider
- **Enhanced Error Handling**: Better handling of API errors with detailed diagnostics
- **Input Validation**: Validates prompts and API keys before requests

## 🛡️ Error Handling

The resume parser includes robust error handling for common issues:

### JSON Parsing
- **Automatic JSON Extraction**: Extracts JSON from markdown code blocks or mixed text
- **Truncation Handling**: Detects and fixes unterminated strings from truncated responses
- **Delimiter Fixing**: Automatically fixes missing commas and brackets
- **Position-Aware Repair**: Uses error positions to apply targeted fixes
- **Fallback Strategies**: Multiple fallback approaches if initial parsing fails

### API Error Handling
- **Detailed Error Messages**: Provides specific API error details for debugging
- **Input Validation**: Validates API keys and prompt inputs
- **Token Limit Management**: Automatically manages token limits per provider
- **Graceful Degradation**: Falls back to partial parsing when possible

## 📁 Data Storage

Parsed resumes are stored in two places:
1. **Neo4j Knowledge Graph**: Structured graph database with relationships
   - Person nodes with personal information
   - Skill nodes with normalization
   - Organization and Position nodes
   - Experience and Education relationships
2. **JSON Files**: Backup in `parsed_resumes/` directory
   - Each resume saved as `<uuid>.json`
   - Contains complete parsed structure
   - Preserved for recovery and analysis

## 🔄 Workflow

1. **Upload Resume**: Upload PDF, DOCX, or TXT file through the Streamlit interface
2. **Extract Text**: Extract raw text from file using appropriate parser (PyPDF2, python-docx, or plain text)
3. **Parse with AI**: Use selected LLM (OpenAI, Anthropic, or Google) to extract structured data
4. **JSON Processing**: Automatically extract and fix JSON from LLM response, handling common errors
5. **Validate Data**: Validate parsed data against Pydantic schema (ResumeData)
6. **Save to Graph**: Store in Neo4j knowledge graph with relationships
7. **Backup to JSON**: Save parsed data as JSON file in `parsed_resumes/` directory
8. **Explore**: Query and explore the knowledge graph through chat interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Neo4j for graph database
- Streamlit for web interface
- OpenAI, Anthropic, Google for AI capabilities

## 📞 Support

For questions or support, please open an issue on GitHub.
