---
page_type: sample
languages:
- python
products:
- azure-machine-learning
description: Running a AutoML Regression in pipeline using SDK. Schedule it.
---

# Use SDK to run AutoML Regression in pipeline
This sample explains how to run an AutoML `Regression` job in pipeline using SDK.

Please find the sample defined in [automl-regression-house-pricing-in-pipeline.ipynb](automl-regression-house-pricing-in-pipeline.ipynb).

# Setup schedule for automl job
- Wrap the `automl.regression()` or `automl.classification()` or `automl.forecasting()` function in a custom `pipeline()`.
- Create an object on the custom pipeline() one, where pass in necessary required params for inner jobs to run - including automl.
- Submit the automl job on wrapped custom pipeline function object.
- Create a job schedule on the wrapped custom pipeline function object.

# Reference
- Great number of automl samples wrapped in a pipeline - available at [1h_automl_in_pipeline](https://github.com/Azure/azureml-examples/tree/main/sdk/python/jobs/pipelines/1h_automl_in_pipeline)
- [Azureml schedule jobs](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-schedule-pipeline-job?view=azureml-api-2&tabs=python#create-a-schedule) documentation
