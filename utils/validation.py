from utils.IModel import IModel

def model_is_valid(model: IModel) -> bool:
    return (
        not model.loss_func == None
        and not model.optimizer_func == None
        and not model.epochs == None
        and not model.batch_size == None
        and not model.lr == None
        and not model.optimizer == None
        and not model.network == None
    )