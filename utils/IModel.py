from torch.nn import Module
import uuid

class IModel:
    id = str(uuid.uuid4())
    loss_func = None
    optimizer_func = None
    epochs = None
    batch_size = None
    lr = None

    optimizer = None
    training_transforms = None
    validation_transforms = None

    network: Module = None