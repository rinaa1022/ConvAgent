"""
Resume Knowledge Graph Manager
Manages resume data in Neo4j knowledge graph
"""

from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
from unified_schema import (
    Person, Organization, Position, Skill, Location,
    Experience, Education, UnifiedRelationships, UnifiedNodeLabels,
    normalize_skill_name, get_skill_aliases, expand_skill_name
)
from collections import Counter
import json
import uuid

# Semantic & lexical similarity
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

_sbert_model = None 


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
                "CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
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

            if person_data.get('projects'):
                self._create_person_projects(session, person_id, person_data['projects'])

        return person_id
    
    def _create_person_skills(self, session, person_id: str, skills: List[Dict[str, Any]]):
        """Create skill nodes and relationships for a person"""
        for skill_data in skills:
            raw_name = skill_data.get('name', '')
            expanded = expand_skill_name(raw_name)

            for skill_name in expanded:
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
    
    def _create_person_projects(self, session, person_id: str, projects: List[Dict[str, Any]]):
        for idx, proj in enumerate(projects):
            title = proj.get("title", "") or proj.get("name", "")
            description = proj.get("description")
            technologies = proj.get("technologies", [])

            # Make sure each project has a unique id
            proj_id = proj.get("id") or f"{person_id}:proj:{idx}"

            session.run("""
                MERGE (pr:Project {id: $proj_id})
                SET pr.title        = $title,
                    pr.description  = $description,
                    pr.technologies = $technologies
                WITH pr
                MATCH (p:Person {id: $person_id})
                MERGE (p)-[:HAS_PROJECT]->(pr)
            """,
            proj_id=proj_id,
            title=title,
            description=description,
            technologies=technologies,
            person_id=person_id)

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
    
    def get_person_projects(self, person_id: str) -> List[Dict[str, Any]]:
        """
        Get all projects for a person.

        Assumed schema:
        (p:Person {id: ...})-[:HAS_PROJECT]->(proj:Project)

        And each Project node may have:
        - title
        - description
        - technologies  (either list or comma-separated string)
        - start_date / end_date (optional)
        """
        with self.driver.session() as session:
            result = session.run("""
                MATCH (p:Person {id: $person_id})-[:HAS_PROJECT]->(proj:Project)
                RETURN
                    proj.title        AS title,
                    proj.description  AS description,
                    proj.technologies AS technologies,
                    proj.start_date   AS start_date,
                    proj.end_date     AS end_date
                ORDER BY start_date
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


    # Job KG methods
    def upsert_jobs(self, jobs: List[Dict[str, Any]]) -> None:
        """
        Upsert a batch of jobs into the KG.

        Each job dict is expected to have at least:
        - job_id, company, role, location, apply_url, age, category, source
        Optionally:
        - skills_required: List[str] of skill names (will be normalized)
        """
        if not jobs:
            return

        prepared_jobs = []
        for job in jobs:
            job_id = job.get("job_id")
            if not job_id:
                continue

            skills_raw = job.get("skills_required") or []
            normalized_skills = []
            for s in skills_raw:
                if not s:
                    continue
                normalized_skills.append(normalize_skill_name(str(s)))

            # inside prepared_jobs building:
            prepared_jobs.append(
                {
                    "job_id": job_id,
                    "company": job.get("company", ""),
                    "role": job.get("role", ""),
                    "location": job.get("location", ""),
                    "apply_url": job.get("apply_url", ""),
                    "age": job.get("age", ""),
                    "category": job.get("category", ""),
                    "source": job.get("source", ""),
                    "skills_required": normalized_skills,
                    "description": job.get("description", ""),
                }
            )


        if not prepared_jobs:
            return

        query = """
        UNWIND $jobs AS job
        WITH job
        WHERE job.job_id IS NOT NULL AND job.job_id <> ""
        MERGE (j:Job {id: job.job_id})
        SET j.title      = job.role,
            j.company    = job.company,
            j.location   = job.location,
            j.apply_url  = job.apply_url,
            j.age        = job.age,
            j.category   = job.category,
            j.source     = job.source,
            j.description = job.description

        // Organization that posts the job
        FOREACH (comp IN CASE WHEN job.company IS NULL OR job.company = "" THEN [] ELSE [job.company] END |
            MERGE (o:Organization {name: comp})
            ON CREATE SET o.type = 'Company'
            MERGE (o)-[:POSTS]->(j)
        )

        // Location of the job
        FOREACH (loc IN CASE WHEN job.location IS NULL OR job.location = "" THEN [] ELSE [job.location] END |
            MERGE (l:Location {name: loc})
            MERGE (j)-[:LOCATED_AT]->(l)
        )

        WITH j, job
        UNWIND coalesce(job.skills_required, []) AS skill_name
        WITH j, skill_name
        WHERE skill_name IS NOT NULL AND trim(skill_name) <> ""
        MERGE (s:Skill {name: skill_name})
        MERGE (j)-[:REQUIRES_SKILL]->(s)
        """

        with self.driver.session() as session:
            session.run(query, jobs=prepared_jobs)
    
    
    def _build_resume_text_for_sbert(self, person_id: str) -> str:
        """
        Build a rich plain-text representation of the resume for SBERT/TF-IDF.
        Uses:
          - summary
          - skills
          - ALL experience descriptions
          - ALL project descriptions (if available)
        """
        person = self.get_person_by_id(person_id) or {}

        # ----- summary -----
        summary = person.get("summary") or ""

        # ----- skills -----
        skills = person.get("skills") or []
        skills_part = ""
        if skills:
            skills_part = "Skills: " + ", ".join(skills)

        # ----- experiences -----
        exps = self.get_person_experiences(person_id)
        exp_lines = []
        for e in exps:
            company = e.get("company") or ""
            position = e.get("position") or ""
            desc = e.get("description") or ""
            loc = e.get("location") or ""

            header_parts = []
            if position:
                header_parts.append(position)
            if company:
                header_parts.append(f"at {company}")
            if loc:
                header_parts.append(f"in {loc}")
            header = " ".join(header_parts)

            if desc:
                exp_lines.append(f"{header}. {desc}" if header else desc)
            elif header:
                exp_lines.append(header)

        exp_part = "\n".join(exp_lines)

        # ----- projects -----
        projects = self.get_person_projects(person_id)
        project_lines = []
        for pr in projects:
            title = pr.get("title") or ""
            desc = pr.get("description") or ""
            techs = pr.get("technologies")

            # technologies might be list OR string
            if isinstance(techs, list):
                tech_str = ", ".join(str(t) for t in techs if t)
            else:
                tech_str = str(techs) if techs else ""

            header = f"Project: {title}" if title else "Project"

            pieces = [header]
            if tech_str:
                pieces.append(f"Tech: {tech_str}")
            if desc:
                pieces.append(desc)

            project_lines.append(". ".join(pieces))

        project_part = "\n".join(project_lines)

        # ----- combine everything -----
        parts = [
            summary,
            skills_part,
            exp_part,
            project_part,
        ]
        return "\n".join(p for p in parts if p.strip())


    def _get_sbert_model(self):
        """ SBERT model once."""
        global _sbert_model
        if _sbert_model is not None:
            return _sbert_model
        if SentenceTransformer is None:
            return None
       
        _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return _sbert_model

    def _build_job_text_for_sbert(self, job: Dict[str, Any]) -> str:
        """Plain-text representation of a job for SBERT / TF-IDF."""
        title = job.get("title", "") or ""
        company = job.get("company", "") or ""
        location = job.get("location", "") or ""
        desc = job.get("description", "") or ""
        skills = job.get("required_skills") or []

        parts = [
            title,
            company,
            location,
            desc,
            "Skills: " + ", ".join(skills) if skills else "",
        ]
        return ". ".join(p for p in parts if p)

    def _sbert_scores(self, resume_text: str, job_texts: List[str]) -> List[float]:
        """Cosine similarity between resume and job texts using SBERT."""
        if not job_texts:
            return []
        model = self._get_sbert_model()
        if model is None or cosine_similarity is None:
            return [0.0] * len(job_texts)

        texts = [resume_text] + job_texts
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        res_vec = embeddings[0:1]
        job_vecs = embeddings[1:]
        sims = cosine_similarity(res_vec, job_vecs)[0]
        return sims.tolist()

    def _tfidf_scores(self, resume_text: str, job_texts: List[str]) -> List[float]:
        """Cosine similarity using TF-IDF over the small candidate set."""
        if not job_texts:
            return []
        if TfidfVectorizer is None or cosine_similarity is None:
            return [0.0] * len(job_texts)

        docs = [resume_text] + job_texts
        vec = TfidfVectorizer(stop_words="english")
        mat = vec.fit_transform(docs)
        sims = cosine_similarity(mat[0:1], mat[1:])[0]
        return sims.tolist()

    def recommend_jobs_for_person(
        self,
        person_id: str,
        limit: int = 5,
        source: str | None = None,
        category: str | None = None,
        location: str | None = None,
    ) -> dict:
        """
        Single-batch recommendation:
        - Fetch up to 50 jobs from Neo4j that overlap in skills.
        - Re-rank them using coverage + SBERT + TF-IDF.
        - Prefer jobs whose location matches the user's query.
        - Return the top `limit` jobs.
        - Also compute missing skills per job and high-level skill suggestions.
        """
        cypher_limit = 50

        query = """
        MATCH (p:Person {id: $person_id})-[:HAS_SKILL]->(rs:Skill)
        WITH p, collect(DISTINCT toLower(rs.name)) AS resumeSkills

        MATCH (j:Job)
        OPTIONAL MATCH (j)-[:REQUIRES_SKILL]->(js:Skill)
        WITH j, resumeSkills, collect(DISTINCT js) AS allJobSkills
        WITH j, resumeSkills, allJobSkills,
             [s IN allJobSkills WHERE toLower(s.name) IN resumeSkills] AS matchingSkills

        WITH j, matchingSkills, allJobSkills,
             size(matchingSkills) AS skill_overlap,
             size(allJobSkills)   AS total_required
        WHERE skill_overlap > 0

        OPTIONAL MATCH (j)<-[:POSTS]-(company:Organization)
        OPTIONAL MATCH (j)-[:LOCATED_AT]->(locNode:Location)
        WITH j, company, locNode, matchingSkills, allJobSkills, skill_overlap, total_required
        WHERE ($source IS NULL OR j.source = $source)
          AND ($category IS NULL OR j.category = $category)

        WITH j, company, locNode, matchingSkills, allJobSkills, skill_overlap, total_required,
            CASE
                WHEN total_required = 0 THEN 0.0
                ELSE toFloat(skill_overlap) / total_required
            END AS coverage,
            CASE
                WHEN $location IS NULL THEN 1
                WHEN j.location IS NOT NULL AND toLower(j.location) CONTAINS toLower($location) THEN 2
                WHEN locNode IS NOT NULL AND toLower(locNode.name) CONTAINS toLower($location) THEN 2
                ELSE 0
            END AS location_score

        RETURN
            coalesce(j.id, j.job_id, id(j))            AS job_id,
            coalesce(j.title, j.role, 'Untitled')      AS title,
            coalesce(company.name, 'Unknown')          AS company,
            coalesce(locNode.name, 'Unknown')          AS location,
            coalesce(j.employment_type, 'Internship')  AS employment_type,
            [s IN matchingSkills | s.name]             AS matching_skills,
            [s IN allJobSkills | s.name]               AS required_skills,
            skill_overlap                              AS matching_skill_count,
            total_required                             AS total_skill_required,
            coverage                                   AS coverage,
            location_score                             AS location_score,
            j.apply_url                                AS apply_url,
            j.source                                   AS source,
            j.category                                 AS category,
            j.description                              AS description
        ORDER BY
            location_score DESC,
            skill_overlap DESC,
            coverage DESC,
            title
        LIMIT $cypher_limit
        """

        with self.driver.session() as session:
            records = session.run(
                query,
                person_id=person_id,
                source=source,
                category=category,
                location=location,
                cypher_limit=cypher_limit,
            )
            jobs = [r.data() for r in records]

        if not jobs:
            return {
                "selected_jobs": [],
                "meta": {
                    "total_considered": 0,
                    "top_skills_from_jobs": [],
                    "improvement_skills": [],
                },
            }

        # ---- SBERT + TF-IDF reranking ----
        try:
            # Build a single resume text from the KG
            resume_text = self._build_resume_text_for_sbert(person_id)

            # Build text for each job
            job_texts = [self._build_job_text_for_sbert(j) for j in jobs]

            sbert_scores = self._sbert_scores(resume_text, job_texts)
            tfidf_scores = self._tfidf_scores(resume_text, job_texts)

            for j, s_score, t_score in zip(jobs, sbert_scores, tfidf_scores):
                j["sbert_score"] = float(s_score)
                j["tfidf_score"] = float(t_score)
                coverage = float(j.get("coverage", 0.0))
                j["combined_score"] = (
                    0.20 * coverage +
                    0.50 * s_score +
                    0.30 * t_score
                )
        except Exception as e:
            print(f"[UnifiedNeo4jManager] SBERT/TF-IDF ranking failed: {e}")
            for j in jobs:
                j["combined_score"] = float(j.get("coverage", 0.0))

        # ---- compute missing skills ----
        for j in jobs:
            required = j.get("required_skills") or []
            matching = j.get("matching_skills") or []

            matching_norm = {normalize_skill_name(s) for s in matching}
            missing = []
            for s in required:
                norm = normalize_skill_name(s)
                if norm not in matching_norm:
                    missing.append(s)
            j["missing_skills"] = missing

        # ---- final ranking: location preference + combined score ----
        jobs.sort(
            key=lambda x: (
                x.get("location_score", 0),       # 2 = matches query, 1 = no filter, 0 = mismatch
                x.get("combined_score", 0.0),
            ),
            reverse=True,
        )

        selected_jobs = jobs[:limit]

        # meta: high-level skill signals
        top_skills_from_jobs = self._top_skills_from_jobs(jobs, top_k=5)
        improvement_skills = self._skills_to_improve_from_jobs(
            person_id=person_id,
            jobs=jobs,
            top_k=5,
        )

        meta = {
            "total_considered": len(jobs),
            "top_skills_from_jobs": top_skills_from_jobs,
            "improvement_skills": improvement_skills,
        }

        return {
            "selected_jobs": selected_jobs,
            "meta": meta,
        }

    def get_resume_text_for_person(self, person_id: str) -> str:
        """
        Build a plain-text representation of the person's resume from the KG.
        """
        person = self.get_person_by_id(person_id)
        if not person:
            return ""

        skills = person.get("skills", [])
        companies = person.get("companies", [])
        institutes = person.get("institutes", [])

        parts = [
            person.get("summary", ""),
            "Skills: " + ", ".join(skills),
            "Companies: " + ", ".join(companies),
            "Education: " + ", ".join(institutes),
        ]
        return "\n".join([p for p in parts if p])

    def _top_skills_from_jobs(self, jobs: list[dict], top_k: int = 5) -> list[str]:
        """
        From up to 50 jobs, compute the most frequent required skills.
        """
        counter = Counter()
        for job in jobs[:50]:
            for s in job.get("required_skills") or []:
                if not s:
                    continue
                norm = normalize_skill_name(str(s))
                counter[norm] += 1

        return [skill for skill, _ in counter.most_common(top_k)]

    def _skills_to_improve_from_jobs(
        self,
        person_id: str,
        jobs: list[dict],
        top_k: int = 5,
    ) -> list[str]:
        """
        Suggest skills that are frequently required by the jobs
        but NOT already present on the person's resume.
        """
        resume_skill_rows = self.get_person_skills(person_id)
        resume_skills = {
            normalize_skill_name(row["skill"])
            for row in resume_skill_rows
            if row.get("skill")
        }

        counter = Counter()
        for job in jobs[:50]:
            for s in job.get("required_skills") or []:
                if not s:
                    continue
                norm = normalize_skill_name(str(s))
                if not norm or norm in resume_skills:
                    continue
                counter[norm] += 1

        return [skill for skill, _ in counter.most_common(top_k)]
