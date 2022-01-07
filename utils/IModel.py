from torch.nn import Module

class IModel(Module):
    loss_func = None
    optimizer_func = None
    epochs = None
    batch_size = None
    lr = None

    optimizer = None
    training_transforms = None
    validation_transforms = None