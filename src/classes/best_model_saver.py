import math
import torch as th


class BestModelSaver:
    def __init__(self, save_path):
        self.best_metric = -math.inf
        self.save_path = save_path

    def __call__(self, metric, model, optimizer, epoch):
        if metric >= self.best_metric:
            self.best_metric = metric
            th.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, self.save_path)
