from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow
import os

# If in azureml compute or job, mlflow tracking URI value is already set in env variable
print("MLFLOW_TRACKING_URI:", os.getenv("MLFLOW_TRACKING_URI"))
print("tracking:", mlflow.get_tracking_uri())

# # If its external compute than azureml workspace, mlflow tracking uri needs to be set
# ## Point to ml workspace
# ml_client = MLClient.from_config(DefaultAzureCredential())
# ## Fetch the workspace resource and read its MLflow tracking URI
# ws = ml_client.workspaces.get(ml_client.workspace_name)
# ## Set the MLflow tracking URI
# mlflow.set_tracking_uri(ws.mlflow_tracking_uri)

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

    for run in child_runs:
        print(f"\trun_id={run.info.run_id}\tstatus={run.info.status}")

        # # Uncomment below to debug specific run - just in case more verbose info needed
        # print("run\t", run)

print("Done")


