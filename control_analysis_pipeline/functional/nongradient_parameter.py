import torch


class NongradParameter:
    def __init__(self, data: torch.tensor = None,
                 lb=-100.0, ub=100.0, precision=1.0,
                 trainable=True):
        self.data = data
        self.trainable = trainable
        self.lb = lb
        self.ub = ub
        self.precision = precision

    def set(self, value: torch.tensor):
        self.data = value

    def get(self):
        return self.data

    def __str__(self):
        out = f"{self.data}"
        return out

    def __repr__(self):
        report = (f"(Data: {self.data}\n"
                  f" LB: {self.lb}\n"
                  f" UB: {self.ub}\n"
                  f" Precision: {self.precision}\n"
                  f" Trainable: {self.trainable})")
        return report

    def __gt__(self, other):
        return self.data > other

    def __lt__(self, other):
        return self.data < other

    def __ge__(self, other):
        return self.data >= other

    def __le__(self, other):
        return self.data <= other
