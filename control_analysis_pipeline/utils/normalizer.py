import torch


class TorchNormalizer:
    def __init__(self, num_of_normalizers):
        self.means = torch.zeros((num_of_normalizers,))
        self.stds = torch.zeros((num_of_normalizers,))

    def fit(self, x):
        """
        Learn transformation.
        :param x: np.array (N x num_of_normalizers)
        :return:
        """
        self.means = torch.mean(x, dim=0)
        self.stds = torch.std(x, dim=0)

    def transform(self, x):
        """
        Normalize data based on trained transformation.
        :param x: np.array (N x num_of_normalizers)
        :return: normalized points (N x num_of_normalizers)
        """
        x = (x - self.means) / self.stds
        return x

    def inverse_transform(self, x):
        """
        Denormalize data based on inverse of the trained transformation.
        :param x: np.array (N x num_of_normalizers)
        :return: denormalized points (N x num_of_normalizers)
        """
        x = x * self.stds + self.means
        return x

    def fit_transform(self, x):
        """
        Learn transformation and transform data.
        :param x: np.array (N x num_of_normalizers)
        :return: normalized points (N x num_of_normalizers)
        """
        self.fit(x)
        x = self.transform(x)
        return x

    def get_json_repr(self):
        return {
            "means": self.means.tolist(),
            "stds": self.stds.tolist()
        }
    