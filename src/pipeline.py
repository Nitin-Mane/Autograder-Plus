# src/pipeline.py (Modified)
import time
from pathlib import Path
from .modules.ingestion import Ingestor
from .modules.static_analyzer import StaticAnalyzer
from .modules.dynamic_analyzer import DynamicAnalyzer
# --- MODIFIED: Import new engines ---
from .modules.embedding_engine import EmbeddingEngine
from .modules.feedback_engine import FeedbackEngine
# --- END MODIFIED ---
from .modules.feedback_generator import FeedbackGenerator
from .modules.analytics_engine import AnalyticsEngine

class Pipeline:
    # --- MODIFIED: Add execution_level to __init__ ---
    def __init__(self, config_path_str: str, submissions_path_str: str, 
                 output_path_str: str, execution_level: str = 'full'):
        """
        Initializes the Autograder pipeline with paths, modules, and execution level.
        """
        print("[PIPELINE] Initializing Autograder Pipeline...")
        
        self.config_path = Path(config_path_str)
        self.submissions_path = Path(submissions_path_str)
        self.output_path = Path(output_path_str)
        self.execution_level = execution_level.lower() # Store the level

        # Initialize base modules that always run
        self.ingestor = Ingestor()
        self.static_analyzer = StaticAnalyzer()
        
        # --- MODIFIED: Conditionally initialize engines based on execution level ---
        # This saves memory and load time if a stage is skipped.
        self.dynamic_analyzer = DynamicAnalyzer() if self.execution_level in ["dynamic", "embedding", "full"] else None
        self.embedding_engine = EmbeddingEngine() if self.execution_level in ["embedding", "full"] else None
        self.feedback_engine = FeedbackEngine() if self.execution_level == "full" else None
        
        self.feedback_generator = FeedbackGenerator()
        self.analytics_engine = AnalyticsEngine()
        
        print(f"[PIPELINE] All necessary modules for level '{self.execution_level}' initialized.")
    # --- END MODIFIED ---

    def _get_assignment_id_from_submissions(self, submissions_data: list) -> str:
        # This method remains unchanged
        if submissions_data and submissions_data[0].get('config'):
            return submissions_data[0]['config'].get('assignment_id', 'unknown_assignment')
        return "unknown_assignment"

    # --- MODIFIED: Conditional logic inside this method ---
    def _process_single_submission(self, submission_data: dict) -> dict:
        """
        Processes a single student submission through the configured analysis stages.
        """
        student_id = submission_data.get('student_id', 'UnknownStudent')
        print(f"\n--- Processing Submission for: {student_id} ---")

        # Initialize analysis structure if not present
        if 'analysis' not in submission_data:
            submission_data['analysis'] = {}
            
        # Stage 1: Static Analysis (always runs)
        analyzed_submission = self.static_analyzer.analyze(submission_data)

        # Stage 2: Dynamic Analysis (runs for 'dynamic', 'embedding', and 'full')
        if self.dynamic_analyzer: # Check if the engine was initialized
            analyzed_submission = self.dynamic_analyzer.analyze(analyzed_submission)
        
        # Stage 3: Embedding Generation (runs for 'embedding' and 'full')
        if self.embedding_engine: # Check if the engine was initialized
            analyzed_submission = self.embedding_engine.analyze(analyzed_submission)

        # Stage 4: Generative Feedback (runs only for 'full')
        if self.feedback_engine: # Check if the engine was initialized
            analyzed_submission = self.feedback_engine.analyze(analyzed_submission)

        print(f"--- Finished Processing for: {student_id} ---")
        return analyzed_submission
    # --- END MODIFIED ---


    def run(self):
        """
        Executes the full grading pipeline based on the configured level.
        """
        # --- This part remains mostly the same, with one conditional change for analytics ---
        start_time = time.time()
        print("\n[PIPELINE] Starting grading process...")

        # 1. Ingestion
        raw_submissions_data = self.ingestor.load_submissions(
            str(self.config_path), str(self.submissions_path)
        )
        if not raw_submissions_data:
            print("[PIPELINE] No submissions found or loaded. Exiting.")
            return
        print(f"[PIPELINE] Ingested {len(raw_submissions_data)} submissions.")

        # 2. Per-Submission Analysis
        processed_submissions = []
        for submission_data_item in raw_submissions_data:
            try:
                processed_submission = self._process_single_submission(submission_data_item)
                processed_submissions.append(processed_submission)
            except Exception as e:
                student_id = submission_data_item.get('student_id', 'UnknownStudent')
                print(f"[PIPELINE] ERROR: Unhandled exception while processing submission for {student_id}: {e}")
                processed_submissions.append({
                    "student_id": student_id,
                    "error_processing": str(e),
                    "config": submission_data_item.get('config', {}), 
                    "analysis": {} 
                })

        assignment_id_for_report = self._get_assignment_id_from_submissions(processed_submissions)
        
        # 3. Aggregated Feedback Generation (always runs, content depends on what was analyzed)
        if processed_submissions:
            self.feedback_generator.generate_all_reports(
                processed_submissions, str(self.output_path), assignment_id_for_report
            )
        else:
            print("[PIPELINE] No submissions processed, skipping feedback report.")

        # 4. Instructor Analytics 
        # --- MODIFIED: Make analytics conditional on embedding generation ---
        print("\n[PIPELINE] Generating instructor analytics...")
        if processed_submissions and self.execution_level in ["embedding", "full"]:
            self.analytics_engine.generate_report(
                processed_submissions, str(self.output_path), assignment_id_for_report
            )
        else:
            print("[PIPELINE] Skipping analytics generation (requires 'embedding' or 'full' level).")
        # --- END MODIFIED ---
            
        # 5. Completion
        end_time = time.time()
        total_time = end_time - start_time
        print(f"\n[PIPELINE] Grading process completed in {total_time:.2f} seconds.")
        print(f"All tasks finished. Check '{self.output_path}' for reports.")