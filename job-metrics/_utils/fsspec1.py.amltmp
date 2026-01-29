from azure.ai.ml import MLClient
from azureml.fsspec import AzureMachineLearningFileSystem
from azure.identity import DefaultAzureCredential

mlclient = MLClient.from_config(credential=DefaultAzureCredential())

subscription_id = mlclient.subscription_id
resource_group = mlclient.resource_group_name
workspace_name = mlclient.workspace_name
datastore_name = "workspaceartifactstore"
path_in_datastore = "ExperimentRun/dcid.khaki_raisin_brh43krrpn_0/"
uri_path = f"azureml://subscriptions/{subscription_id}/resourceGroups/{resource_group}/workspaces/{workspace_name}/datastores/{datastore_name}/paths/{path_in_datastore}"

print(uri_path)

# Instatiate the filesystem using uri
fs = AzureMachineLearningFileSystem(uri = uri_path)

# List files and directories at the root of the specified path
print(fs.ls())

# Recursively list all files and directories
for dirpath, dirnames, filenames in fs.walk(""):
    print(f"Directory: {dirpath}")
    for dirname in dirnames:
        print(f"  Subdirectory: {dirname}")
    for filename in filenames:
        print(f"  File: {filename}")

# Add a conditional check to see if any child level folder or file contains the name `model`
for dirpath, dirnames, filenames in fs.walk(""):
    for dirname in dirnames:
        if "model" in dirname.lower():
            print(f"Found 'model' in directory: {dirpath}/{dirname}")
    for filename in filenames:
        if "model" in filename.lower():
            print(f"Found 'model' in file: {dirpath}/{filename}")


# glob for recursive listing
paths = fs.glob("**/*model*")
for path in paths:
    print("path: ", path)

if paths:
    print("this run has model association")
else:
    print("this run has no model association")
