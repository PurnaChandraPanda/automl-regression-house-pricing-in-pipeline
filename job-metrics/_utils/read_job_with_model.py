
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from mlflow.tracking import MlflowClient
from azure.core.exceptions import HttpResponseError


def get_root_run_id(run_id: str) -> str:
    mfc = MlflowClient()
    run = mfc.get_run(run_id)
    return run.data.tags.get("mlflow.rootRunId") or run.info.run_id


def model_outputs_from_job(job_entity):
    hits = []
    for name, out in (job_entity.outputs or {}).items():
        t = getattr(out, "type", None)
        if t in ("mlflow_model", "custom_model"):
            hits.append({"output_name": name, "type": t, "path": getattr(out, "path", None)})
    return hits


def model_outputs(job):
    hits = []
    for k, out in (job.outputs or {}).items():
        t = getattr(out, "type", None)
        print("t: ", t)
        if t in ("mlflow_model", "custom_model"):
            hits.append((k, t, getattr(out, "path", None)))
    return hits


def get_job_model_outputs(ml_client: MLClient, job_name: str):
    job = ml_client.jobs.get(job_name)
    # print("job: ", job)
    
    print("job.name:", job.name)
    print("job.type:", getattr(job, "type", None))
    print("job.display_name:", getattr(job, "display_name", None))
    print("has outputs:", bool(getattr(job, "outputs", None)))
    print("outputs:", getattr(job, "outputs", None))

    outputs = job.outputs or {}
    
    children = list(ml_client.jobs.list(parent_job_name=job_name))
    print("Child job count:", len(children))
    
    found = []
    supported = []
    unsupported = []
    for c in children:
        cj = None
        try:
            cj = ml_client.jobs.get(c.name)
            supported.append(cj)
        except HttpResponseError as e:
            # Known pattern for internal jobs
            if "JobNotSupported" in str(e) or "not supported in this API version" in str(e):
                unsupported.append(c.name)
                continue
            raise

    print("Supported children:", len(supported))
    print("Unsupported children:", len(unsupported))

    for job in supported:
        # print(job)
        
        hits = model_outputs(job)
        if hits:
            print("\n✅ Model outputs for", job.name, hits)
    
    print("Child jobs with model outputs:", len(found))
    
    for job_name, hits in found[:20]:
        print("\n✅", job_name)
        for h in hits:
            print("  ", h)

    return found, outputs

# --- usage ---
ml_client = MLClient.from_config(DefaultAzureCredential())
runid = "khaki_raisin_brh43krrpn_0"

root = get_root_run_id(runid)   # usually equals the AML job name for the parent/root
model_outs, all_outs = get_job_model_outputs(ml_client, root)

print("Root/job:", root)
print("All job outputs keys:", list(all_outs.keys()))
print("Model-typed outputs:", model_outs)
