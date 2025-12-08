"""
Resume Knowledge Graph Schema
Defines the ontology for resume data in Neo4j knowledge graph
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import re

class SkillCategory(Enum):
    TECHNICAL = "Technical"
    SOFT = "Soft"
    LANGUAGE = "Language"
    TOOL = "Tool"
    FRAMEWORK = "Framework"
    PLATFORM = "Platform"

class ExperienceLevel(Enum):
    ENTRY = "Entry"
    JUNIOR = "Junior"
    MID = "Mid"
    SENIOR = "Senior"
    LEAD = "Lead"
    PRINCIPAL = "Principal"
    MANAGEMENT = "Management"


# Core Entities
@dataclass
class Person:
    """Person entity (from resumes)"""
    id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    summary: Optional[str] = None

@dataclass
class Organization:
    """Unified organization entity (companies, institutes)"""
    id: str
    name: str
    type: str  # "Company", "Institute", "Government", etc.
    industry: Optional[str] = None
    size: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    description: Optional[str] = None

@dataclass
class Position:
    """Position entity (job titles, roles, academic positions)"""
    id: str
    title: str
    level: Optional[ExperienceLevel] = None
    department: Optional[str] = None
    category: Optional[str] = None  # "Engineering", "Marketing", etc.
    description: Optional[str] = None

@dataclass
class Skill:
    """Unified skill entity"""
    id: str
    name: str
    category: SkillCategory = SkillCategory.TECHNICAL
    proficiency_level: Optional[str] = None
    years_experience: Optional[int] = None
    aliases: List[str] = field(default_factory=list)  # Alternative names

@dataclass
class Location:
    """Geographic location entity"""
    id: str
    name: str
    country: Optional[str] = None
    state: Optional[str] = None
    city: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    timezone: Optional[str] = None

@dataclass
class Experience:
    """Work experience entity"""
    id: str
    position: str
    company: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    skills_used: List[str] = field(default_factory=list)

@dataclass
class Education:
    """Education entity"""
    id: str
    institute: str
    degree: str
    major: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    courses: List[str] = field(default_factory=list)

@dataclass
class Project:
    """Project entity"""
    id: str
    title: str
    description: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


# Unified Relationship Types
class UnifiedRelationships:
    """Constants for unified knowledge graph relationships"""
    
    # Person relationships
    PERSON_HAS_SKILL = "HAS_SKILL"
    PERSON_HAS_EXPERIENCE = "HAS_EXPERIENCE"
    PERSON_HAS_EDUCATION = "HAS_EDUCATION"
    PERSON_LOCATED_IN = "LOCATED_IN"
    PERSON_HAS_PROJECT = "HAS_PROJECT" 
    
    # Organization relationships
    ORG_LOCATED_IN = "LOCATED_IN"
    ORG_IN_INDUSTRY = "IN_INDUSTRY"
    
    # Position relationships
    POSITION_REQUIRES_SKILL = "REQUIRES_SKILL"
    POSITION_AT_LEVEL = "AT_LEVEL"
    POSITION_IN_DEPARTMENT = "IN_DEPARTMENT"
    
    # Experience relationships
    EXPERIENCE_AT_COMPANY = "AT_COMPANY"
    EXPERIENCE_IN_POSITION = "IN_POSITION"
    EXPERIENCE_USED_SKILL = "USED_SKILL"
    EXPERIENCE_LOCATED_IN = "LOCATED_IN"
    
    # Education relationships
    EDUCATION_AT_INSTITUTE = "AT_INSTITUTE"
    EDUCATION_FOR_DEGREE = "FOR_DEGREE"
    EDUCATION_MAJOR_IN = "MAJOR_IN"
    
    # Skill relationships
    SKILL_RELATED_TO = "RELATED_TO"
    SKILL_ALIAS_OF = "ALIAS_OF"
    SKILL_IN_CATEGORY = "IN_CATEGORY"

# Unified Node Labels
class UnifiedNodeLabels:
    """Constants for unified node labels in Neo4j"""
    
    # Core entities
    PERSON = "Person"
    ORGANIZATION = "Organization"
    POSITION = "Position"
    SKILL = "Skill"
    LOCATION = "Location"
    
    # Supporting entities
    PROJECT = "Project"
    CERTIFICATION = "Certification"
    LANGUAGE = "Language"
    
    # Classification entities
    INDUSTRY = "Industry"
    DEPARTMENT = "Department"
    DEGREE = "Degree"
    MAJOR = "Major"
    COURSE = "Course"

# Skill Ontology - Common skill mappings
SKILL_ONTOLOGY = {
    # Programming Languages
    "python": ["python programming", "python3", "python 3", "py"],
    "javascript": ["js", "ecmascript", "node.js", "nodejs"],
    "java": ["java programming", "jdk"],
    "typescript": ["ts", "typescript programming"],
    "c": ["c", "c language", "objective c"],
    "c++": ["cpp", "c plus plus"],
    "c#": ["csharp", "c sharp"],
    
    # Web Frameworks
    "react": ["reactjs", "react.js", "react js"],
    "angular": ["angularjs", "angular.js"],
    "vue": ["vuejs", "vue.js"],
    "django": ["django framework"],
    "flask": ["flask framework"],
    "express": ["express.js", "expressjs"],
    
    # Databases
    "sql": ["sql database", "structured query language"],
    "mysql": ["mysql database"],
    "postgresql": ["postgres", "postgresql database"],
    "mongodb": ["mongo", "mongodb database"],
    "redis": ["redis database"],
    
    # Cloud Platforms
    "aws": ["amazon web services", "amazon aws"],
    "azure": ["microsoft azure", "azure cloud"],
    "gcp": ["google cloud platform", "google cloud"],
    
    # Tools and Technologies
    "docker": ["docker containerization"],
    "kubernetes": ["k8s", "kubernetes orchestration"],
    "git": ["git version control"],
    "jenkins": ["jenkins ci/cd"],
    "terraform": ["terraform iac"],
    
    # Soft Skills
    "leadership": ["team leadership", "leading teams"],
    "communication": ["verbal communication", "written communication"],
    "problem solving": ["analytical thinking", "critical thinking"],
    "teamwork": ["collaboration", "team collaboration"],
}

def get_skill_aliases(skill_name: str) -> List[str]:
    """Get aliases for a skill name"""
    skill_lower = skill_name.lower().strip()
    return SKILL_ONTOLOGY.get(skill_lower, [])

def normalize_skill_name(skill_name: str) -> str:
    """Normalize skill name to canonical form"""
    if not skill_name:
        return ""
    
    skill_lower = skill_name.lower().strip()

    # If it's exactly a canonical key
    if skill_lower in SKILL_ONTOLOGY:
        return skill_lower.title()

    # Check aliases
    for canonical, aliases in SKILL_ONTOLOGY.items():
        if skill_lower in aliases:
            return canonical.title()
    
    return skill_lower.title()

def expand_skill_name(skill_name: str) -> List[str]:
    """
    Expand composite skill labels like 'C/C++' into multiple canonical skills.
    Example:
      'C/C++' or 'C / C++'  -> ['C', 'C++']
      otherwise             -> [<single normalized skill>]
    """
    if not skill_name:
        return []

    skill_lower = skill_name.lower().strip()

    # Special handling for C / C++
    # Matches: "c/c++", "c / c++", "c++/c", "c++ / c" (any spacing, any case handled by lower())
    if re.fullmatch(r"c\s*/\s*c\+\+|c\+\+\s*/\s*c", skill_lower):
        return ["C", "C++"]

    # single normalized skill
    return [normalize_skill_name(skill_name)]