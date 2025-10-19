import subprocess
import sys
import os

# --- Configuration ---
# Ensure these file names match your actual files
FEATURE_ENGINEERING_SCRIPT = 'feature_engineering.py'
TRAIN_AGENT_SCRIPT = 'train_agent.py'
EVALUATE_AGENT_SCRIPT = 'evaluate_agent.py'

# This should be the path to the python executable in your virtual environment
# On Windows, it's typically 'venv\\Scripts\\python.exe'
PYTHON_EXECUTABLE = os.path.join('venv', 'Scripts', 'python.exe')


def run_script(script_name):
    """A helper function to run a python script and check for errors."""
    print(f"\n--- Running: {script_name} ---")
    
    # Check if the script file exists
    if not os.path.exists(script_name):
        print(f"[!] ERROR: Script not found: {script_name}")
        sys.exit(1) # Exit the program with an error code
        
    # Run the script using the python from our virtual environment
    result = subprocess.run([PYTHON_EXECUTABLE, script_name], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[!] ERROR running {script_name}.")
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"--- Finished: {script_name} successfully ---")
        print(result.stdout)


# --- Main execution block ---
if __name__ == "__main__":
    print("==============================================")
    print("=== Starting AI Trading Bot Project Workflow ===")
    print("==============================================")

    # Step 1: Create the master feature dataset
    run_script(FEATURE_ENGINEERING_SCRIPT)
    
    # Step 2: Train the AI agent
    # NOTE: This step can take a very long time!
    run_script(TRAIN_AGENT_SCRIPT)
    
    # Step 3: Evaluate the trained agent
    run_script(EVALUATE_AGENT_SCRIPT)

    print("\n=============================================")
    print("=== Project Workflow Completed Successfully ===")
    print("=============================================")