import uuid
import os
import datetime

class ModelMetadata:
    serialization_attributes = [
        "Id",
        "Accuracy",
        "Epochs",
        "Learning_rate",
        "Loss_func",
        "Optimizer_func",
        "Timestamp"
    ]

    def __init__(self, model, accuracy, config):
        self.id = str(uuid.uuid4())
        self.model_path = os.path.join(config.MODELS_PATH, f"{self.id}.pth")
        self.class_path = os.path.join(config.MODELS_PATH, f"{self.id}.py")
        self.accuracy = accuracy
        self.epochs = config.EPOCHS
        self.learning_rate = config.LR
        self.loss_func = type(model.loss_func).__name__
        self.optimizer_func = model.optimizer_func.__name__
        self.timestamp = datetime.datetime.now()
    
    def __str__(self) -> str:
        parameters = map(str,[
            self.id,
            self.accuracy,
            self.epochs,
            self.learning_rate,
            self.loss_func,
            self.optimizer_func,
            self.timestamp
        ])

        return ",".join(parameters)