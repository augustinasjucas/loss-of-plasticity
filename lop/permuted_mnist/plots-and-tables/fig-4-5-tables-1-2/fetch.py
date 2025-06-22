import wandb
import pandas as pd
import pickle
import concurrent.futures
from tqdm import tqdm

entity = "augustinasj"
project = "mnist-after-hyperparam"


def fetch_run_data(run):
    """Fetches and processes data for a single run."""
    try:
        # Get full history and convert to pandas DataFrame
        history = run.scan_history()
        df = pd.DataFrame(list(history))
        
        # Handle possible None group
        group = run.group or 'ungrouped'
        
        return (
            group,
            run.id,
            {
                "run_history": df,
                "run_name": run.name,
                "config": run.config
            }
        )
    except Exception as e:
        print(f"Error processing run {run.id}: {str(e)}")
        return (None, None, None)

def main():

    # Initialize WandB API
    api = wandb.Api(timeout=60)
    
    # Get all runs from the project
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs in project {project}")
    
    # Create dictionary to store results
    results_dict = {}
    
    # Process runs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Create futures
        futures = {executor.submit(fetch_run_data, run): run for run in runs}
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(runs), desc="Processing runs"):
            group, run_id, data = future.result()
            
            # Skip failed runs
            if None in (group, run_id) or data is None:
                continue
            
            # Update results dictionary
            if group not in results_dict:
                results_dict[group] = {}
            results_dict[group][run_id] = data
    
    # Save results to pickle file
    with open('SAVED_POST_HYPERPARAM.pkl', 'wb') as f:
        pickle.dump(results_dict, f)
    
    print("Data successfully saved to SAVED_POST_HYPERPARAM.pkl")

if __name__ == "__main__":
    main()
