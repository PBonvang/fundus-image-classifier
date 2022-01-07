import uuid
import os
import datetime
import Model

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

    def __init__(self, model, accuracy, config):
        self.id = str(uuid.uuid4())
        self.model_path = os.path.join(config.MODELS_PATH, f"{self.id}.pth")
        self.class_path = os.path.join(config.MODELS_PATH, f"{self.id}.py")
        self.accuracy = accuracy
        self.epochs = config.EPOCHS
        self.batch_size = config.FEATURE_EXTRACTION_BATCH_SIZE
        self.learning_rate = config.LR
        self.loss_func = type(Model.LOSS_FUNC).__name__
        self.optimizer_func = Model.OPTIMIZER_FUNC.__name__
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