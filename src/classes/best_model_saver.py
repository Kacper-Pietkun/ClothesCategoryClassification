import torch as th


class BestModelSaver:
    def __init__(self, save_path):
        self.best_val_accuracy = -1
        self.save_path = save_path

    def __call__(self, val_accuracy, model, optimizer, epoch):
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            th.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, self.save_path)
