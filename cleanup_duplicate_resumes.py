"""
Cleanup script to remove duplicate resume JSON files
Identifies duplicates based on resume content and optionally syncs with Neo4j
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Set
from unified_neo4j_manager import UnifiedNeo4jManager
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

def get_resume_signature(resume_data: Dict) -> str:
    """Create a unique signature for a resume based on name and email only"""
    personal_info = resume_data.get('personal_info', {})
    name = personal_info.get('name', '').strip().lower()
    email = personal_info.get('email', '').strip().lower()
    
    # Only use name and email for duplicate detection
    # Both must be present and match for it to be considered a duplicate
    if not name or not email:
        # If name or email is missing, use UUID to keep them separate
        return f"unique_{resume_data.get('id', 'unknown')}"
    
    # Create signature from name and email only
    return f"{name}|{email}"

def load_resume_files(resumes_dir: str = "parsed_resumes") -> Dict[str, List[Dict]]:
    """Load all resume JSON files and group by signature"""
    if not os.path.exists(resumes_dir):
        print(f"Directory {resumes_dir} does not exist!")
        return {}
    
    resumes_by_signature = defaultdict(list)
    
    for filename in os.listdir(resumes_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(resumes_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                resume_data = json.load(f)
                
            signature = get_resume_signature(resume_data)
            resume_data['_filename'] = filename
            resume_data['_filepath'] = filepath
            resumes_by_signature[signature].append(resume_data)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    return resumes_by_signature

def get_neo4j_person_ids(manager: UnifiedNeo4jManager) -> Set[str]:
    """Get all person IDs currently in Neo4j"""
    try:
        people = manager.get_all_people()
        return {person.get('id') for person in people if person.get('id')}
    except Exception as e:
        print(f"Error fetching Neo4j person IDs: {e}")
        return set()

def find_duplicates(dry_run: bool = True, check_neo4j: bool = True):
    """Find and optionally remove duplicate resume files"""
    print("=" * 60)
    print("Resume Duplicate Cleanup Script")
    print("=" * 60)
    print()
    
    # Load all resumes grouped by signature
    resumes_by_signature = load_resume_files()
    
    if not resumes_by_signature:
        print("No resume files found!")
        return
    
    print(f"Found {sum(len(group) for group in resumes_by_signature.values())} total resume files")
    print(f"Grouped into {len(resumes_by_signature)} unique resumes")
    print()
    
    # Get Neo4j person IDs if checking
    neo4j_ids = set()
    if check_neo4j:
        print("Checking Neo4j for existing person IDs...")
        try:
            manager = UnifiedNeo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
            neo4j_ids = get_neo4j_person_ids(manager)
            print(f"Found {len(neo4j_ids)} person IDs in Neo4j")
            manager.close()
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Continuing without Neo4j check...")
            check_neo4j = False
        print()
    
    duplicates_to_remove = []
    kept_files = []
    
    # Analyze each group
    for signature, resumes in resumes_by_signature.items():
        if len(resumes) == 1:
            # No duplicates for this signature
            kept_files.append(resumes[0])
            continue
        
        print(f"Duplicate group found ({len(resumes)} files):")
        personal_info = resumes[0].get('personal_info', {})
        name = personal_info.get('name', 'Unknown')
        email = personal_info.get('email', 'N/A')
        print(f"  ✓ Same Name: {name}")
        print(f"  ✓ Same Email: {email}")
        print(f"  Files:")
        
        # Sort by parsed_at date (newest first) and presence in Neo4j
        def sort_key(resume):
            resume_id = resume.get('id', '')
            in_neo4j = resume_id in neo4j_ids
            parsed_at = resume.get('parsed_at', '')
            # Prioritize: in Neo4j > newer date
            return (not in_neo4j, parsed_at)
        
        sorted_resumes = sorted(resumes, key=sort_key)
        
        # Keep the first one (prefer in Neo4j, then newest)
        keep_resume = sorted_resumes[0]
        kept_files.append(keep_resume)
        
        for i, resume in enumerate(sorted_resumes):
            resume_id = resume.get('id', '')
            filename = resume['_filename']
            parsed_at = resume.get('parsed_at', 'N/A')
            in_neo4j = resume_id in neo4j_ids if check_neo4j else False
            
            marker = "✓ KEEP" if i == 0 else "✗ DELETE"
            neo4j_status = " (in Neo4j)" if in_neo4j else " (not in Neo4j)"
            
            print(f"    {marker} {filename}")
            print(f"        ID: {resume_id}")
            print(f"        Parsed: {parsed_at}{neo4j_status}")
        
        # Mark others for deletion
        for resume in sorted_resumes[1:]:
            duplicates_to_remove.append(resume)
        
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Unique resumes: {len(kept_files)}")
    print(f"Duplicate files to remove: {len(duplicates_to_remove)}")
    print()
    
    if duplicates_to_remove:
        print("Files to be removed:")
        for resume in duplicates_to_remove:
            print(f"  - {resume['_filename']} (ID: {resume.get('id', 'N/A')})")
        print()
        
        if dry_run:
            print("🔍 DRY RUN MODE - No files will be deleted")
            print("Run with dry_run=False to actually delete files")
        else:
            # Actually delete the files
            deleted_count = 0
            for resume in duplicates_to_remove:
                try:
                    os.remove(resume['_filepath'])
                    deleted_count += 1
                    print(f"✓ Deleted: {resume['_filename']}")
                except Exception as e:
                    print(f"✗ Error deleting {resume['_filename']}: {e}")
            
            print()
            print(f"✅ Successfully deleted {deleted_count} duplicate files")
    else:
        print("✅ No duplicates found!")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    dry_run = True
    check_neo4j = True
    
    if len(sys.argv) > 1:
        if "--delete" in sys.argv or "-d" in sys.argv:
            dry_run = False
        if "--no-neo4j" in sys.argv:
            check_neo4j = False
    
    if dry_run:
        print("⚠️  Running in DRY RUN mode. Use --delete flag to actually remove files.")
        print()
    
    find_duplicates(dry_run=dry_run, check_neo4j=check_neo4j)

