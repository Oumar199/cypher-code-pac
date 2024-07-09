from cypher_code_package import wandb


def download_checkpoint(project, artifact_path, key):

    wandb.login(key=key)

    run = wandb.init(project=project)

    artifact = run.use_artifact(artifact_path, type="dataset")

    artifact_dir = artifact.download()

    wandb.finish()

    return artifact_dir
