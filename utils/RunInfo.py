from datetime import datetime
from math import inf
import os
import uuid
import pandas as pd

from utils.IModel import IModel


class RunInfo:
    epochs_run: int = 0
    samples_seen: int = 0
    steps_taken: int = 0

    best_training_loss: float = inf
    best_training_accuracy: float = 0

    best_validation_loss: float = inf
    best_validation_accuracy: float = 0

    test_accuracy = 0.0

    best_net = None

    def __init__(self,
                 model: IModel,
                 run_path: str,
                 ds_size: int,
                 id: str = str(uuid.uuid4()),
                 name: str = None,
                 base_model_path=None):
        self.id = id
        self.model_id = model.id
        self.name = name
        self.run_path = run_path
        self.checkpoint_path = os.path.join(self.run_path, "checkpoints")

        self.ds_size = ds_size
        self.batch_size = model.batch_size
        self.learning_rate = model.lr
        self.optimizer_func = model.optimizer_func.__name__

        self.base_model_path = base_model_path
        self.timestamp = datetime.now()

    def set_test_accuracy(self, accuracy: float):
        self.test_accuracy = accuracy

    def save_to_csv(self, file: str):
        df: pd.DataFrame = None

        if os.path.exists(file):
            df = pd.read_csv(file)
        else:
            df = pd.DataFrame(columns=[
                "Model_Id",
                "Run_Id",
                "Name",
                "Accuracy",
                "Best_training_loss",
                "Best_training_acc",
                "Best_validation_loss",
                "Best_validation_acc",
                "Dataset_size",
                "Epochs",
                "Samples_seen",
                "Steps_taken",
                "Base",
                "Timestamp"
            ])

        df.loc[len(df.index)] = [
            self.model_id,
            self.id,
            self.name,
            self.test_accuracy,
            self.best_training_loss,
            self.best_training_accuracy,
            self.best_validation_loss,
            self.best_validation_accuracy,
            self.ds_size,
            self.epochs_run,
            self.samples_seen,
            self.steps_taken,
            self.base_model_path,
            self.timestamp
        ]

        df.to_csv(file, index = False)
