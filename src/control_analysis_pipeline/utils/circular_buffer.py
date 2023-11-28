import torch


class CircularBuffer:
    def __init__(self, data_buffer: torch.tensor, dim=0):
        self.data = None
        self.start = 0
        self.dim = 0
        self.length = 0
        self.reset(data_buffer, dim)

    def add(self, data_entry):
        if self.data.dim() == 1:
            self.data[self.start] = data_entry
        else:
            temp = self.data.select(self.dim, self.start)
            temp[:] = data_entry
        self.start += 1
        if self.start >= self.length:
            self.start = 0

    def reset(self, data_buffer, dim):
        self.data = data_buffer
        self.start = 0
        self.dim = dim
        self.length = self.data.shape[dim]

    def get(self):
        return torch.cat((
            torch.index_select(
                self.data,
                dim=self.dim,
                index=torch.tensor(range(self.start, self.length), dtype=torch.int)),
            torch.index_select(
                self.data,
                dim=self.dim,
                index=torch.tensor(range(self.start), dtype=torch.int)),
        ), dim=self.dim)


if __name__ == "__main__":
    buff = CircularBuffer(torch.zeros((1, 5)), 1)
    buff.add(10.0)
    buff.add(1.0)
    print(buff.get())
    print(torch.all(torch.isclose(buff.get(), torch.tensor([10., 1., 0., 0., 0.]))))

    print()
    buff.reset(torch.zeros((2, 3, 4)), 1)
    print(buff.get())
    buff.add(torch.tensor([[5.0, 2, 3, 5], [1, 2, 3, 5]]))
    # buff.add(torch.tensor([1, 2, 3]))
    # buff.add(torch.tensor([1, 2, 3]))
    # buff.add(torch.tensor([1, 2, 3]))
    # buff.add(torch.tensor([1, 2, 3]))
    # buff.add(torch.tensor([7, 8, 9]))
    print(buff.get())

    # print()
    # buff.reset(torch.zeros((3, 5)), 1)
    # print(buff.get())
    # buff.add(torch.tensor([0, 0, 0]))
    # buff.add(torch.tensor([0, 0, 0]))
    # buff.add(torch.tensor([1, 9, 3]))
    # buff.add(torch.tensor([2, 9, 3]))
    # buff.add(torch.tensor([3, 9, 3]))
    # buff.add(torch.tensor([4, 9, 3]))
    # buff.add(torch.tensor([5, 9, 3]))
    # print("------------------")
    # print()
    # print(buff.data)
    # print()
    # print(buff.get())
