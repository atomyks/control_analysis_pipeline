import torch
import torch.nn as nn
from control_analysis_pipeline.helpers.circular_buffer import CircularBuffer
import inspect
from typing import Callable


class RegressorFactory(nn.Module):
    """
    Similar to https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html
    """

    def __init__(self, batch_size=1, num_actions=1, num_states=1, action_history_size=1, state_history_size=1):
        super(RegressorFactory, self).__init__()

        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_states = num_states
        self.regressor_library = []
        self.action_history_size = action_history_size
        self.state_history_size = state_history_size

        self.action_history: CircularBuffer or None = None  # (BATCH x HISTORY x NUM_INPUTS)
        self.state_history: CircularBuffer or None = None  # (BATCH x HISTORY x NUM_INPUTS)

        self.reset_history()

    def forward(self, u_now: torch.tensor, y_prev: torch.tensor) -> torch.tensor:
        """
        :param u_now: (BATCH x NUM_ACTIONS)
        :param y_prev: (BATCH x NUM_STATES)
        :return: computed regressors (BATCH x NUM_STATES)
        """

        # Fix dimensions
        u_now = u_now.reshape((self.batch_size, self.num_actions))
        y_prev = y_prev.reshape((self.batch_size, self.num_states))

        # add new history entry
        self.action_history.add(u_now)
        self.state_history.add(y_prev)

        history_a = self.action_history.get()
        history_s = self.state_history.get()

        # compute regressor vector
        regressor_vector = []

        for regressor in self.regressor_library:
            in_s = []
            in_a = []
            fun, a_defs, s_defs = regressor
            if a_defs is not None:
                for j, a_ in enumerate(a_defs):
                    history_idx, action_idx = a_
                    in_a.append(history_a[..., history_idx - 1, action_idx].reshape((self.batch_size, 1)))
            if s_defs is not None:
                for j, s_ in enumerate(regressor[2]):
                    history_idx, state_idx = s_
                    in_s.append(history_s[..., history_idx - 1, state_idx].reshape((self.batch_size, 1)))
            regressor_vector.append(fun(in_a, in_s))

        return torch.cat(regressor_vector, dim=1)

    def reset_regressor_library(self):
        self.regressor_library = []

    def set_history(self, action_history: torch.tensor, state_history: torch.tensor) -> None:
        current_action_history_shape = torch.tensor([self.batch_size, self.action_history_size, self.num_actions])
        current_state_history_shape = torch.tensor([self.batch_size, self.state_history_size, self.num_states])
        if not torch.equal(torch.tensor(action_history.shape), current_action_history_shape):
            raise ValueError('dimension mismatch')
        if not torch.equal(torch.tensor(state_history.shape), current_state_history_shape):
            raise ValueError('dimension mismatch')

        self.action_history = CircularBuffer(action_history, dim=1)
        self.state_history = CircularBuffer(state_history, dim=1)

    def reset_history(self):
        # (BATCH x HISTORY x NUM_INPUTS)
        self.action_history = CircularBuffer(torch.zeros(self.batch_size, self.action_history_size, self.num_actions),
                                             dim=1)

        # (BATCH x HISTORY x NUM_INPUTS)
        self.state_history = CircularBuffer(torch.zeros(self.batch_size, self.action_history_size, self.num_actions),
                                            dim=1)

    def add(self,
            fun: Callable,
            a_defs: list or None = None,
            s_defs: list or None = None
            ) -> None:
        """
        :param fun: Lambda function that generates regressor element with desired operation.`
                    Should have two actions: a, s (action, state)
                    Inputs should be treated as lists even if there is only one element.
                    Note: Since we will use this in learning all functions should use pytorch.
                    Example: lambda a, s: a[0] * a[1] + s[0]

        :param a_defs: List of tuples, where each tuple is of the form (lag, index) for the actions used  in "fun", in order.
                       The length of the list should be equal to A+1 where A is the number of actions used in "fun".
        :param s_defs: List of tuples, where each tuple is of the form (lag, index) for the states used in "fun", in order.
                       The length of the list should be equal to S+1 where S is the number of states used in "fun".

        Usage example 1:
            # To add the regressor a[k, 0] * a[k-5, 2] + s[k-4, 0] we first define the corresponding
            # a_def and s_def
            # (0, 0) defines that a[0] is action number 0 and history element k (most recent one).
            # (-5, 2) defines that a[1] is action number 2 and history element k-5
            a_def = [(0, 0), (-5, 2)]

            # (-4, 0) defines that s[0] is state number 0 and history element k-4
            s_def = [(-4, 0)]

            # create regressor that uses
            f = lambda a, s: a[0] * a[1] + s[0]
            reg.add(f, a_def, s_def)

        Usage example 1:
            # (0, 0) defines that a[0] is action number 0 and history element k (most recent one).
            a_def = [(0, 0)]s
            f = lambda a, s: torch.pow(a[0], 2.0)
            reg.add(f, a_def)
            f = lambda a, s: torch.pow(a[0], 3.0)
            reg.add(f, a_def)
        """
        if s_defs is None:
            s_defs = []
        self.regressor_library.append((fun, a_defs, s_defs))

    def __str__(self):
        out_text = ""
        out_text += "---------<Regressor Library>---------\n"
        for i in range(len(self.regressor_library)):
            regressor_text = ""
            found_start = False
            regressor_ = self.regressor_library[i]
            for ch in inspect.getsource(regressor_[0]):
                if found_start:
                    regressor_text += ch
                if ch == ':':
                    found_start = True
            # First argument
            # for a_ in regressor_[1]:
            if regressor_[1] is not None:
                for j, a_ in enumerate(regressor_[1]):
                    regressor_text = regressor_text.replace(f"a[{j}]", f"a{a_[1]}[k{a_[0]}]")
            if regressor_[2] is not None:
                for j, s_ in enumerate(regressor_[2]):
                    regressor_text = regressor_text.replace(f"s[{j}]", f"s{s_[1]}[k{s_[0]}]")
            regressor_text = regressor_text.replace(f"k0", f"k")
            out_text += f" R{i}: " + regressor_text
        return out_text


if __name__ == "__main__":
    batch_size = 3
    num_actions = 2
    num_states = 2
    action_history_size = 4
    state_history_size = 4

    reg = RegressorFactory(
        batch_size=batch_size,
        num_actions=num_actions,
        num_states=num_states,
        action_history_size=action_history_size,
        state_history_size=state_history_size
    )

    reg.set_history(action_history=torch.rand((batch_size, action_history_size, num_actions)),
                    state_history=torch.rand((batch_size, state_history_size, num_states))
                    )

    # a - action (BATCH x HISTORY x num_actions)
    # s - state (BATCH x HISTORY x num_states)

    # (0, 0) defines that a[0] is action number 0 and history element k (most recent one).
    a_def = [(-2, 0)]
    f = lambda a, s: torch.pow(a[0], 2.0)
    reg.add(f, a_def)
    a_def = [(-2, 1)]
    f = lambda a, s: torch.pow(a[0], 3.0)
    reg.add(f, a_def)


    print(reg)

    regressor_vector = reg(torch.rand((batch_size, num_actions)),
                           torch.rand((batch_size, num_states)))

    print(torch.pow(reg.action_history.get()[:, -3, 0], 2.0))
    print(torch.pow(reg.action_history.get()[:, -3, 1], 3.0))
    print(regressor_vector)
