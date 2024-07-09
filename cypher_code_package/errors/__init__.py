class EmptyArtifactException(Exception):
    def __init__(
        self, error="No wandb's artifact path was provided!! Please provide one."
    ):

        print(error)
