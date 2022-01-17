import datetime

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

    def __init__(self, model: IModel):
        self.id = model.id
        self.epochs = model.epochs
        self.batch_size = model.batch_size
        self.learning_rate = model.lr
        self.loss_func = type(model.loss_func).__name__
        self.optimizer_func = model.optimizer_func.__name__
        self.timestamp = datetime.datetime.now()

    def set_accuracy(self, accuracy: float):
        self.accuracy = accuracy
    
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