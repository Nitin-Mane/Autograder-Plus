# src/modules/ingestion.py (Modified)
import json
from pathlib import Path

class Ingestor:
    def load_submissions(self, config_path: str, submissions_path: str) -> list:
        """
        Loads submissions from a specified directory.
        Handles two structures:
        1. Directory-based: submissions_path/student_id/code.py
        2. File-based:     submissions_path/student_id.py
        """
        print("[INGESTOR] Loading assignment config and submissions...")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                assignment_config = json.load(f)
        except FileNotFoundError:
            print(f"[INGESTOR] ERROR: Assignment config file not found at '{config_path}'")
            return []
        except json.JSONDecodeError:
            print(f"[INGESTOR] ERROR: Assignment config file at '{config_path}' is not valid JSON.")
            return []

        submissions = []
        root_path = Path(submissions_path)
        
        if not root_path.is_dir():
            print(f"[INGESTOR] ERROR: Submissions path '{submissions_path}' is not a valid directory.")
            return []

        print(f"[INGESTOR] Scanning for submissions in: {root_path}")
        for item_path in root_path.iterdir():
            submission_data = None
            
            # --- Case 1: Item is a directory (e.g., ./submissions/hw2/student101/) ---
            if item_path.is_dir():
                student_id = item_path.name
                try:
                    # Find the first .py file inside the student's directory
                    code_file_path = next(item_path.glob('*.py'))
                    print(f"  [INGESTOR] Found directory submission for '{student_id}' at '{code_file_path}'")
                    
                    with open(code_file_path, 'r', encoding='utf-8') as f:
                        code = f.read()

                    submission_data = {
                        "student_id": student_id,
                        "code_path": str(code_file_path),
                        "code": code,
                        "config": assignment_config,
                        "analysis": {}
                    }
                except StopIteration:
                    print(f"  [INGESTOR] WARNING: No .py file found in directory for student: {student_id}")

            # --- Case 2: Item is a Python file directly in the submissions folder ---
            elif item_path.is_file() and item_path.suffix == '.py':
                # Use the filename (without extension) as the student_id
                student_id = item_path.stem 
                code_file_path = item_path
                print(f"  [INGESTOR] Found direct file submission for '{student_id}' at '{code_file_path}'")
                
                try:
                    with open(code_file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        
                    submission_data = {
                        "student_id": student_id,
                        "code_path": str(code_file_path),
                        "code": code,
                        "config": assignment_config,
                        "analysis": {}
                    }
                except Exception as e:
                     print(f"  [INGESTOR] ERROR: Could not read file '{code_file_path}'. Details: {e}")


            # If a valid submission was processed, add it to the list
            if submission_data:
                submissions.append(submission_data)

        if not submissions:
             print(f"[INGESTOR] WARNING: No valid submissions found in '{submissions_path}'. Please check directory structure and file names.")
             
        return submissions