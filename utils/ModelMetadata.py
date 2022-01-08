import uuid
import os
import datetime
import config

from utils.IModel import IModel

class ModelMetadata:
    serialization_attributes = [
        "Id",
        "Accuracy",
        "Epochs",
        "Batch_size",
        "Learning_rate",
        "Loss_func",
        "Optimizer_func",
        "Timestamp"
    ]

    def __init__(self, model: IModel, accuracy):
        self.id = str(uuid.uuid4())
        self.model_path = os.path.join(config.TRAINED_MODELS_PATH, f"{self.id}.pth")
        self.class_path = os.path.join(config.MODELS_PATH, f"{self.id}.py")
        self.accuracy = accuracy
        self.epochs = model.epochs
        self.batch_size = model.batch_size
        self.learning_rate = model.lr
        self.loss_func = type(model.loss_func).__name__
        self.optimizer_func = model.optimizer_func.__name__
        self.timestamp = datetime.datetime.now()
    
    def __str__(self) -> str:
        parameters = map(str,[
            self.id,
            self.accuracy,
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.loss_func,
            self.optimizer_func,
            self.timestamp
        ])

        return ",".join(parameters)