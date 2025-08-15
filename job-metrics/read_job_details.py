from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import mlflow

# Point to ml workspace
ml_client = MLClient.from_config(DefaultAzureCredential())

# Root job run id
root_run_id = "" # Place the actual job run id e.g. olive_eye_8t4wmfjv20

# Read ml job details
run = ml_client.jobs.get(name=root_run_id)

# Access the current job run details
with mlflow.start_run(run_id=root_run_id) as run:
    _run = mlflow.get_run(run.info.run_id)
    print(_run.info.run_id, _run.info.status, _run.info.run_name)
    experiment_id = _run.info.experiment_id

# Search for child runs on the experiment_id basis
child_runs = mlflow.search_runs(
    experiment_ids=[experiment_id],
    output_format="list",
)
print("child_runs ", len(child_runs))

# Filter for child runs for specific root_run_id
specific_child_runs = [
    r
    for r in child_runs
    if r.data.to_dictionary().get("tags").get("mlflow.rootRunId") == root_run_id
]
print("specific_child_runs ", len(specific_child_runs))

for child_run in specific_child_runs:
    # print(type(child_run.data)) #<class 'mlflow.entities.run_data.RunData'>
    # Read the mlflow run data
    _run_data = child_run.data.to_dictionary()
    # Read metrics of the mlflow run, where its non-empty dictionary
    _run_metrics = _run_data.get("metrics")
    if _run_metrics:
        print("Metrics for run:", child_run.info.run_id, child_run.info.status)
        for metric_name, metric_value in _run_metrics.items():
            print(f"  {metric_name}: {metric_value}")

