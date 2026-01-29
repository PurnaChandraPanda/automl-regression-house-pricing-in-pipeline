from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.fsspec import AzureMachineLearningFileSystem

import mlflow
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


# If in azureml compute or job, mlflow tracking URI value is already set in env variable
print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))
print("tracking:", mlflow.get_tracking_uri())


# If its external compute than azureml workspace, mlflow tracking uri needs to be set
## Point to ml workspace
_ml_client = MLClient.from_config(credential = DefaultAzureCredential())
# ## Fetch the workspace resource and read its MLflow tracking URI
# ws = _ml_client.workspaces.get(_ml_client.workspace_name)
# ## Set the MLflow tracking URI
# mlflow.set_tracking_uri(ws.mlflow_tracking_uri)


def _get_run_storage_uri(runid: str) -> str:
    """Get the run storage URI path for fsspec access."""
    # workspaceartifactstore: is one of the in-built datastores in AML workspace
    ## its callers need to ensure have at least "storage blob data reader" role on the storage account behind the datastore
    subscription_id = _ml_client.subscription_id
    resource_group = _ml_client.resource_group_name
    workspace_name = _ml_client.workspace_name
    datastore_name = "workspaceartifactstore"
    path_in_datastore = f"ExperimentRun/dcid.{runid}/"
    fs_uri_path = f"azureml://subscriptions/{subscription_id}/resourceGroups/{resource_group}/workspaces/{workspace_name}/datastores/{datastore_name}/paths/{path_in_datastore}"

    # print(fs_uri_path)

    return fs_uri_path


def _run_has_model(run_id: str) -> bool:
    """Check if the given run has model association by checking its artifact storage."""
    ## Case 1: Using fsspec, check the run artifact storage for model files.
    ## Case 2: Using models.list(), check models for job_name (or run_id) association. Model will be associated with artifact storage of the run.
    ### In either case, artifact storage validation for the run is needed.

    # Get the run storage URI path
    uri_path = _get_run_storage_uri(run_id)
    # print(uri_path)

    # Instatiate the filesystem using uri
    fs = AzureMachineLearningFileSystem(uri = uri_path)

    # glob for recursive listing
    paths = fs.glob("**/*model*")
    # for path in paths:
    #     print("path: ", path)

    # Check if any model related files/ folders found. Return its results.
    if paths:
        return True
    else:        
        return False

def main():
    # Lists all experiments visible in the current tracking URI
    experiments = mlflow.search_experiments()   # returns List[Experiment]

    # Print all experiments and their child runs
    for e in experiments:
        print(f"name={e.name}\tid={e.experiment_id}\tlifecycle={e.lifecycle_stage}")

        # Search for child runs on the experiment_id basis
        child_runs = mlflow.search_runs(
            experiment_ids=[e.experiment_id],
            output_format="list",
        )
        print("len(child_runs): ", len(child_runs))
        print("========================================")

        # Parallelize I/O-bound remote checks
        _max_workers = 8 #2
        results = {}
        with ThreadPoolExecutor(max_workers=_max_workers) as pool:
            future_to_runid = {
                pool.submit(_run_has_model, run.info.run_id): run.info.run_id
                for run in child_runs
            }
            for fut in as_completed(future_to_runid):
                run_id = future_to_runid[fut]
                try:
                    results[run_id] = fut.result()
                except Exception:
                    results[run_id] = False

        for run in child_runs:
            # Check if the run has model association
            if results.get(run.info.run_id, False):
                print(f"\trun_id={run.info.run_id}\tstatus={run.info.status}")

            # # Uncomment below to debug specific run - just in case more verbose info needed
            # if run.info.run_id == "khaki_raisin_brh43krrpn_0":
            #     print("run\t", run)

if __name__ == "__main__":
    main()
    print("=======Done=========")


