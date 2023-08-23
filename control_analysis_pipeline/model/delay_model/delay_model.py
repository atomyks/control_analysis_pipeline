import torch
from control_analysis_pipeline.model.model import Model
from control_analysis_pipeline.model.nongradient_parameter import NongradParameter
import queue

class InputDelayModel(Model):
    """
    System input delay module. This can be trained only using non-gradient optimization
    """
    def __init__(self, num_actions=1):
        super(InputDelayModel, self).__init__(num_actions=num_actions)
        self.delay_parm = NongradParameter("delay", 0, 10, 1)
        self.register_nongrad_parameter("delay", self.delay_parm)
        self.batch_size = 1
        self.delay_parm.set(0)

    def init_learning(self, batch_size):
        '''
        Used to initialize the size of the queue. This function is called before the first call of the learn function. 
        :return:
        '''
        self.batch_size = batch_size
        self.reset()

    def forward(self, u_input: torch.tensor):
        if self.delay_parm.get() > 0:
            y_output = self.input_queue.get()
            self.input_queue.put(u_input.reshape((self.batch_size, self.num_actions)))
        else:
            y_output = u_input.reshape((self.batch_size, self.num_actions))
        return y_output

    def reset(self):
        self.input_queue = queue.Queue(maxsize=self.delay_parm.get())
        for i in range(self.delay_parm.get()):
            u = torch.zeros((self.batch_size, self.num_actions), dtype=torch.float64)
            self.input_queue.put(u)
