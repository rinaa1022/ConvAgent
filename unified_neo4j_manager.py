"""
Resume Knowledge Graph Manager
Manages resume data in Neo4j knowledge graph
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from unified_schema import (
    Person, Organization, Position, Skill, Location,
    Experience, Education, UnifiedRelationships, UnifiedNodeLabels,
    normalize_skill_name, get_skill_aliases
)
import json
import uuid

class UnifiedNeo4jManager:
    """Manages knowledge graph for resume data"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """Close the database connection"""
        self.driver.close()
    
    def clear_database(self):
        """Clear all data from the database (for testing)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared")
    
    def create_constraints(self):
        """Create database constraints for better performance"""
        with self.driver.session() as session:
            # Unique constraints
            constraints = [
                "CREATE CONSTRAINT person_id_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT organization_name_unique IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
                "CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE",
                "CREATE CONSTRAINT position_title_unique IF NOT EXISTS FOR (p:Position) REQUIRE p.title IS UNIQUE",
                "CREATE CONSTRAINT location_name_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    print(f"Created constraint: {constraint.split('FOR')[0].strip()}")
                except Exception as e:
                    print(f"Constraint may already exist: {e}")
    
    # Resume-related methods
    def create_person(self, person_data: Dict[str, Any]) -> str:
        """Create a person node from resume data"""
        person_id = person_data.get('id', str(uuid.uuid4()))
        
        with self.driver.session() as session:
            session.run("""
                CREATE (p:Person {
                    id: $person_id,
                    name: $name,
                    email: $email,
                    phone: $phone,
                    linkedin: $linkedin,
                    github: $github,
                    summary: $summary
                })
            """, 
            person_id=person_id,
            name=person_data.get('personal_info', {}).get('name', ''),
            email=person_data.get('personal_info', {}).get('email'),
            phone=person_data.get('personal_info', {}).get('phone'),
            linkedin=person_data.get('personal_info', {}).get('linkedin'),
            github=person_data.get('personal_info', {}).get('github'),
            summary=person_data.get('summary')
            )
            
            # Create skills
            if person_data.get('skills'):
                self._create_person_skills(session, person_id, person_data['skills'])
            
            # Create experiences
            if person_data.get('experience'):
                self._create_person_experiences(session, person_id, person_data['experience'])
            
            # Create education
            if person_data.get('education'):
                self._create_person_education(session, person_id, person_data['education'])
        
        return person_id
    
    def _create_person_skills(self, session, person_id: str, skills: List[Dict[str, Any]]):
        """Create skill nodes and relationships for a person"""
        for skill_data in skills:
            skill_name = normalize_skill_name(skill_data.get('name', ''))
            if not skill_name:
                continue
                
            session.run("""
                MERGE (s:Skill {name: $skill_name})
                ON CREATE SET 
                    s.category = $category,
                    s.proficiency_level = $proficiency
                ON MATCH SET 
                    s.category = COALESCE(s.category, $category)
                WITH s
                MATCH (p:Person {id: $person_id})
                MERGE (p)-[r:HAS_SKILL]->(s)
                ON CREATE SET r.proficiency = $proficiency
            """, 
            skill_name=skill_name,
            category=skill_data.get('category', 'Technical'),
            proficiency=skill_data.get('proficiency'),
            person_id=person_id
            )
    
    def _create_person_experiences(self, session, person_id: str, experiences: List[Dict[str, Any]]):
        """Create experience nodes and relationships for a person"""
        for exp_data in experiences:
            # Create organization
            company_name = exp_data.get('company', '')
            session.run("""
                MERGE (o:Organization {name: $company_name})
                ON CREATE SET 
                    o.type = 'Company',
                    o.location = $location
                ON MATCH SET 
                    o.location = COALESCE(o.location, $location)
            """, 
            company_name=company_name,
            location=exp_data.get('location')
            )
            
            # Create position
            position_title = exp_data.get('position', '')
            session.run("""
                MERGE (p:Position {title: $position_title})
                ON CREATE SET 
                    p.category = 'Professional',
                    p.description = $description
            """, 
            position_title=position_title,
            description=exp_data.get('description')
            )
            
            # Create experience relationship
            session.run("""
                MATCH (person:Person {id: $person_id})
                MATCH (org:Organization {name: $company_name})
                MATCH (pos:Position {title: $position_title})
                CREATE (person)-[:HAS_EXPERIENCE {
                    start_date: $start_date,
                    end_date: $end_date,
                    description: $description,
                    location: $location
                }]->(org)
                MERGE (org)-[:HAS_POSITION]->(pos)
            """, 
            person_id=person_id,
            company_name=company_name,
            position_title=position_title,
            start_date=exp_data.get('dates', {}).get('from_date'),
            end_date=exp_data.get('dates', {}).get('to_date'),
            description=exp_data.get('description'),
            location=exp_data.get('location')
            )
            
            # Create skill relationships for this experience
            for skill_name in exp_data.get('skills_used', []):
                normalized_skill = normalize_skill_name(skill_name)
                session.run("""
                    MERGE (s:Skill {name: $skill_name})
                    WITH s
                    MATCH (pos:Position {title: $position_title})
                    MERGE (pos)-[:REQUIRES_SKILL]->(s)
                """, 
                skill_name=normalized_skill,
                position_title=position_title
                )
    
    def _create_person_education(self, session, person_id: str, education: List[Dict[str, Any]]):
        """Create education nodes and relationships for a person"""
        for edu_data in education:
            institute_name = edu_data.get('institute', '')
            
            # Create institute
            session.run("""
                MERGE (o:Organization {name: $institute_name})
                ON CREATE SET o.type = 'Institute'
            """, institute_name=institute_name)
            
            # Create education relationship
            session.run("""
                MATCH (person:Person {id: $person_id})
                MATCH (institute:Organization {name: $institute_name})
                CREATE (person)-[:HAS_EDUCATION {
                    degree: $degree,
                    major: $major,
                    start_date: $start_date,
                    end_date: $end_date,
                    gpa: $gpa
                }]->(institute)
            """, 
            person_id=person_id,
            institute_name=institute_name,
            degree=edu_data.get('degree'),
            major=json.dumps(edu_data.get('major', [])),
            start_date=edu_data.get('dates', {}).get('from_date'),
            end_date=edu_data.get('dates', {}).get('to_date'),
            gpa=edu_data.get('gpa')
            )
    
    # Query methods for resume data
    def get_person_by_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get a person by their ID"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {id: $person_id})
                OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
                OPTIONAL MATCH (p)-[:HAS_EXPERIENCE]->(org:Organization)
                OPTIONAL MATCH (p)-[:HAS_EDUCATION]->(edu:Organization)
                RETURN p, 
                       collect(DISTINCT s.name) as skills,
                       collect(DISTINCT org.name) as companies,
                       collect(DISTINCT edu.name) as institutes
            """, person_id=person_id)
            
            record = result.single()
            if record:
                person = dict(record['p'])
                person['skills'] = record['skills']
                person['companies'] = record['companies']
                person['institutes'] = record['institutes']
                return person
            return None
    
    def get_all_people(self) -> List[Dict[str, Any]]:
        """Get all people in the database"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person)
                OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
                RETURN p.id as id, p.name as name, count(DISTINCT s) as skill_count
                ORDER BY p.name
            """)
            
            return [dict(record) for record in result]
    
    def get_person_skills(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all skills for a person"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {id: $person_id})-[r:HAS_SKILL]->(s:Skill)
                RETURN s.name as skill, s.category as category, r.proficiency as proficiency
                ORDER BY s.name
            """, person_id=person_id)
            
            return [dict(record) for record in result]
    
    def get_person_experiences(self, person_id: str) -> List[Dict[str, Any]]:
        """Get all work experiences with positions for a person"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {id: $person_id})-[exp:HAS_EXPERIENCE]->(org:Organization)
                OPTIONAL MATCH (org)-[:HAS_POSITION]->(pos:Position)
                RETURN org.name as company, 
                       pos.title as position,
                       exp.start_date as start_date,
                       exp.end_date as end_date,
                       exp.description as description,
                       exp.location as location
                ORDER BY exp.start_date DESC
            """, person_id=person_id)
            
            return [dict(record) for record in result]
    
    def find_skill_usage(self) -> List[Dict[str, Any]]:
        """Get skill usage statistics across all resumes"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
                RETURN s.name as skill, s.category as category, count(DISTINCT p) as person_count
                ORDER BY person_count DESC
            """)
            
            return [dict(record) for record in result]
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        with self.driver.session() as session:
            stats = {}
            node_types = [
                'Person', 'Organization', 'Position', 'Skill', 'Location'
            ]
            
            for node_type in node_types:
                result = session.run(f"MATCH (n:{node_type}) RETURN count(n) as count")
                stats[node_type] = result.single()['count']
            
            # Get relationship counts
            relationship_types = [
                'HAS_SKILL', 'HAS_EXPERIENCE', 'HAS_EDUCATION', 'HAS_POSITION', 'REQUIRES_SKILL'
            ]
            
            for rel_type in relationship_types:
                result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                stats[f"{rel_type}_relationships"] = result.single()['count']
            
            return stats
    
    def test_connection(self) -> bool:
        """Test Neo4j connection"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()['test'] == 1
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

