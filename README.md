# Autograder+: A Multi-Faceted AI Framework for Rich Pedagogical Feedback in Programming Education

Autograder+ is an advanced, command-line-based framework designed to revolutionize the assessment of programming assignments. It moves beyond traditional autograders by integrating a multi-stage analysis pipeline that provides deep, instructional insights for both students and educators. The system combines secure, containerized code execution with state-of-the-art AI, including specialized code embedding models and large language models (LLMs), to deliver comprehensive feedback and powerful classroom analytics.

![UMAP Plot of Student Submissions](reports/hw2/umap_plot.png)
*(Example UMAP visualization showing semantic clusters of student solutions)*

### Core Features

*   **Secure & Flexible Execution:** Utilizes Docker to run student code in isolated, sandboxed environments. Natively supports both full-program scripts and function-only submissions.
*   **Tiered Analysis Pipeline:** Allows users to select the depth of analysis (`dynamic`, `embedding`, `full`) to balance speed with the richness of feedback.
*   **Semantic Code Embeddings:** Employs models like `nomic-ai/nomic-embed-code` to convert code into meaningful vector representations, enabling quantitative analysis of solution similarity.
*   **AI-Generated Pedagogical Feedback:** Leverages powerful LLMs (e.g., Qwen2) served locally via Ollama to provide structured, human-like feedback on debugging, code functionality, and conceptual understanding.
*   **Instructor Analytics Dashboard:** Generates interactive UMAP visualizations that map the entire class's solution space, allowing instructors to visually identify common strategies, shared misconceptions, and successful approaches.
*   **Fine-Tuning Capabilities:** Includes scripts for fine-tuning embedding models using advanced techniques like Multi-Label Supervised Contrastive Learning to make them "correctness-aware."

---

## Installation and Setup Guide

Follow these steps to set up and run the Autograder+ framework on your local machine.

### Step 1: Prerequisites

Before you begin, ensure you have the following software installed and configured:

1.  **Python:** Python 3.9 or higher is recommended.
2.  **Git:** For cloning the repository.
3.  **Docker:** Docker must be installed and the Docker daemon must be running.
    *   [Install Docker Engine](https://docs.docker.com/engine/install/)
    *   **Linux Users:** After installation, you must add your user to the `docker` group to run Docker commands without `sudo`.
        ```bash
        sudo usermod -aG docker $USER
        ```
        **Important:** You need to **log out and log back in** for this change to take effect. You can verify by running `docker ps`, which should execute without a permission error.
4.  **Ollama (for AI Feedback):**
    *   Ollama is required to run the generative LLMs locally. [Install Ollama](https://ollama.com/).
    *   After installation, pull the model(s) you intend to use. For example, to pull a 7-billion parameter Qwen2 instruct model, run:
        ```bash
        ollama pull qwen2:7b-instruct
        ```
    *   Ensure the model tag in `src/modules/feedback_engine.py` (e.g., `OLLAMA_MODEL_ID`) matches a model you have pulled. You can see your local models with `ollama list`.

### Step 2: Clone the Repository

Open your terminal and clone the project repository:

```bash
git clone https://github.com/your-username/autograder-plus.git
cd autograder-plus

(Replace your-username/autograder-plus.git with your actual repository URL.)

### Step 3: Set Up the Python Environment

It is strongly recommended to use a Python virtual environment to manage dependencies.

1.Create a virtual environment:

    
python3 -m venv venv

  

2.Activate the virtual environment:

  On macOS / Linux
  
source venv/bin/activate

  

On Windows:
        
    .\venv\Scripts\activate

      

Install required packages:

The requirements.txt file contains all necessary Python libraries.

    pip install -r requirements.txt

      

You are now ready to use Autograder+!


## Project Structure and File Descriptions

The project is organized into several key directories:

autograder-plus/
├── assignments/            # Assignment configurations
│   └── hwX/
│       └── config.json
├── reports/                # Generated output reports
│   └── hwX/
│       ├── Report_... .md
│       ├── Summary_... .csv
│       └── interactive_embeddings_... .html
├── submissions/            # Student code submissions
│   └── hwX/
│       ├── student_id/
│       │   └── main.py
│       └── student_id.py
├── src/                    # Source code for the autograder
│   └── modules/            # All analysis and generation modules
│       ├── ingestion.py
│       ├── static_analyzer.py
│       ├── dynamic_analyzer.py
│       ├── embedding_engine.py
|       ├── prompt_pool.py
│       ├── feedback_engine.py
│       ├── feedback_generator.py
│       └── analytics_engine.py
│   └── pipeline.py         # Main pipeline orchestrator
├── finetune/               # (Optional) Scripts for model fine-tuning
│   ├── finetune_embeddings.py
│   └── supcon_loss.py
├── main.py                 # Main Command-Line Interface (CLI) entry point
└── requirements.txt        # Python package dependencies



Key File Descriptions

    main.py: The entry point for the application. It uses click to define the command-line interface and its arguments (--level, --config, etc.).

    src/pipeline.py: The central orchestrator. It initializes all the engine modules and runs the analysis pipeline in the correct sequence based on the selected --level.

    src/modules/:

        ingestion.py: Reads config.json and finds/loads student code from the submissions directory. Handles both student_id/code.py and student_id.py formats.

        static_analyzer.py: Performs pre-execution checks on the code using Abstract Syntax Trees (ASTs) to find syntax errors and basic code structures.

        dynamic_analyzer.py: The core execution engine. It uses Docker to securely run student code against test cases and captures the results. It generates a runner.py script on-the-fly to handle different execution_modes.

        embedding_engine.py: Loads a pre-trained or fine-tuned embedding model (e.g., nomic-embed-code) to convert code into semantic vectors.

        feedback_engine.py: Connects to the local Ollama server to send prompts (containing student code, errors, and instructor questions) to a generative LLM (e.g., Qwen2) and parses the structured feedback.

        feedback_generator.py: Consumes the results from all analysis stages and generates the final human-readable reports (aggregated .md and summary .csv).

        analytics_engine.py: Takes the embeddings from all students, performs UMAP dimensionality reduction, and generates the interactive Plotly (.html) visualization.

    finetune/: Contains standalone scripts for advanced users to fine-tune embedding models on custom datasets (e.g., using Multi-Label Supervised Contrastive Learning).

    assignments/: Instructors place their config.json files here to define new assignments.

    submissions/: Student code should be placed here, following the structure defined for the assignment.

    reports/: All output files are saved here by default.
    
    
Usage

Run the autograder from the root directory of the project.
Command Structure
code Bash

    
python main.py grade --level <LEVEL> --assignment-config <PATH> --submissions-dir <PATH> --output-dir <PATH>

  

Examples
Run the Full Pipeline (Default)

This runs all stages, including dynamic tests, embedding generation, and AI feedback generation.
code Bash

    
python main.py grade \
    --assignment-config ./assignments/hw2/config.json \
    --submissions-dir ./submissions/hw2/ \
    --output-dir ./reports/hw2_full_run

  

Run for Analytics Only

This is faster as it skips the slow generative feedback stage. It's perfect for generating the UMAP plot.
code Bash

    
python main.py grade --level embedding \
    --assignment-config ./assignments/hw2/config.json \
    --submissions-dir ./submissions/hw2/ \
    --output-dir ./reports/hw2_embedding_run

  

Run for Quick Correctness Check

This is the fastest mode, running only the classic autograder functionality.
code Bash

    
python main.py grade --level dynamic \
    --assignment-config ./assignments/hw2/config.json \
    --submissions-dir ./submissions/hw2/ \
    --output-dir ./reports/hw2_dynamic_only

  


