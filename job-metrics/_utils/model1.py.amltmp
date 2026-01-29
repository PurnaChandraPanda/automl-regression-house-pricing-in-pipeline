from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

mlclient = MLClient.from_config(credential=DefaultAzureCredential())

# List the models in ml workspace
# models = mlclient.models.list()

# Enumerate the models for model name
for model in mlclient.models.list():
    # print(model.name, model.version)
    # if model.version is not None:
    #     print(mlclient.models.get(model.name, model.version))
    for _model in mlclient.models.list(name=model.name):
        # print(_model.name, _model.version)
        if "/dcid." in _model.path:
            print(_model.name, _model.version, _model.type, _model.path, _model.job_name)

    # if "dcid." in model.path:
    #     print(model.name, model.version, model.type, model.path, model.job_name)

# models = mlclient.models.list(name="dogs_dev")
# for model in models:
#     print(model.name, model.version)
#     print(model)
#     break