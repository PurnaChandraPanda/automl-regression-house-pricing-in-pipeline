
## Find the job id from ml studio
![](../.media/find-job-runid-from-experiment.png)

Replace the `root job id` in script placeholder. Root job id is not mandatory, but would be nice if root is picked.

```python
# Root job run id
root_run_id = "" # Place the actual job run id e.g. olive_eye_8t4wmfjv20

# Read ml job details
run = ml_client.jobs.get(name=root_run_id)

# Access the current job run details
with mlflow.start_run(run_id=root_run_id) as run:
    _run = mlflow.get_run(run.info.run_id)
    print(_run.info.run_id, _run.info.status, _run.info.run_name)
    experiment_id = _run.info.experiment_id
```

## Run the script to load child job metrics

```
conda activate azureml_py310_sdkv2
pip install -U azure-ai-ml
pip install -U azureml-mlflow
cd job-metrics
python read_job_details.py
```
![](../.media/child-job-metrics-read-1.png)
![](../.media/child-job-metrics-read-2.png)

## Reference for mlflow search
[Manage runs with mlflow](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments-mlflow?view=azureml-api-2#query-and-search-experiments)

