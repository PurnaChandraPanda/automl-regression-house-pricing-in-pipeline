import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
root_job = "khaki_raisin_brh43krrpn"
root_run = client.get_run(root_job)
exp_id = root_run.info.experiment_id

df = mlflow.search_runs(
    experiment_ids=[exp_id],
    filter_string=f"tags.mlflow.rootRunId = '{root_job}'"
)
print("Runs under root:", len(df))
print(df[["run_id", "status", "tags.mlflow.runName"]])


def find_mlmodel_files(run_id: str):    
    stack = [""]
    hits = []
    while stack:
        p = stack.pop()
        for fi in client.list_artifacts(run_id, p):
            if fi.is_dir:
                stack.append(fi.path)
            else:
                if fi.path == "MLmodel" or fi.path.endswith("/MLmodel"):
                    hits.append(fi.path)
    return hits

model_runs = []
for rid in df["run_id"].tolist():
    paths = find_mlmodel_files(rid)
    if paths:
        model_runs.append((rid, paths))

print("✅ Runs that contain MLflow model artifacts:", len(model_runs))
for rid, paths in model_runs[:10]:
    print("\nRun:", rid)
    for p in paths:
        print("  MLmodel:", p)
