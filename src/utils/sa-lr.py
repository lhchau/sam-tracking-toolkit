import torch

class StepLR:
    def __init__(self, optimizer, learning_rate: float, total_epochs: int):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = learning_rate
        self.sharpness_history = []

    def __call__(self, sharpness):
        self.sharpness_history.append(sharpness)
        alpha = sharpness / torch.median(torch.tensor(self.sharpness_history))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.base * alpha

    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]