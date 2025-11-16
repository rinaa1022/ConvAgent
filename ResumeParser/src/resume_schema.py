from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class DateRange(BaseModel):
    from_date: Optional[str] = Field(None, description="Start date in YYYY-MM format")
    to_date: Optional[str] = Field(None, description="End date in YYYY-MM format or 'Present'")

class Education(BaseModel):
    institute: str = Field(..., description="Name of the educational institution")
    degree: str = Field(..., description="Degree obtained (e.g., Bachelor's, Master's, PhD)")
    major: List[str] = Field(default_factory=list, description="Major field(s) of study")
    dates: DateRange = Field(..., description="Duration of education")
    courses: List[str] = Field(default_factory=list, description="Relevant courses taken")
    gpa: Optional[str] = Field(None, description="Grade Point Average if mentioned")

class Experience(BaseModel):
    position: str = Field(..., description="Job title or position")
    company: str = Field(..., description="Company or organization name")
    dates: DateRange = Field(..., description="Duration of employment")
    description: str = Field(..., description="Job description and responsibilities")
    skills_used: List[str] = Field(default_factory=list, description="Skills utilized in this role")
    location: Optional[str] = Field(None, description="Work location if mentioned")

class Skill(BaseModel):
    name: str = Field(..., description="Skill name")
    category: str = Field(..., description="Skill category (e.g., Technical, Soft, Language)")
    proficiency: Optional[str] = Field(None, description="Proficiency level if mentioned")

class Project(BaseModel):
    name: str = Field(..., description="Project name")
    description: str = Field(..., description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    dates: Optional[DateRange] = Field(None, description="Project duration if mentioned")
    url: Optional[str] = Field(None, description="Project URL if available")

class Certification(BaseModel):
    name: str = Field(..., description="Certification name")
    issuer: str = Field(..., description="Certifying organization")
    date: Optional[str] = Field(None, description="Certification date")
    expiry: Optional[str] = Field(None, description="Expiry date if applicable")

class ResumeData(BaseModel):
    personal_info: Dict[str, Any] = Field(default_factory=dict, description="Personal information like name, email, phone, address")
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    education: List[Education] = Field(default_factory=list, description="Educational background")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    skills: List[Skill] = Field(default_factory=list, description="Skills and competencies")
    projects: List[Project] = Field(default_factory=list, description="Projects and portfolio")
    certifications: List[Certification] = Field(default_factory=list, description="Certifications and licenses")
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    achievements: List[str] = Field(default_factory=list, description="Notable achievements and awards")
    
    @field_validator('languages', mode='before')
    @classmethod
    def clean_languages(cls, v):
        """Clean languages array to remove null values"""
        if not v:
            return []
        if isinstance(v, list):
            return [str(item) for item in v if item is not None and str(item).strip()]
        return []
    
    @field_validator('achievements', mode='before')
    @classmethod
    def clean_achievements(cls, v):
        """Clean achievements array to remove null values"""
        if not v:
            return []
        if isinstance(v, list):
            return [str(item) for item in v if item is not None and str(item).strip()]
        return []
