import argparse
from control_analysis_pipeline.system.system import System
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from yaml import load
from yaml import Loader

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151 - http://www.mwm.im/lqr-controllers-with-python/
    
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, X, eigVals

def main():
    config = None
    with open("double_int_config.yaml", "r") as stream:
        config = load(stream, Loader=Loader)

    sys = System()
    sys.parse_config(config)

    # Double integrator system
    sys.set_linear_model_matrices(A=torch.tensor([[1, 1], [0, 1]], dtype=torch.float64),
                                  B=torch.tensor([[0], [1]], dtype=torch.float64))

    # Simulate the system closed loop with LQR controller (no delay)
    # Simulation ends when the state is within 1e-15 of the origin
    # Repeat the simulation for 100 times to collect a dataset  
    inputs_arr = []
    outputs_arr = []
    initial_states = []
    
    Q = np.eye(2)
    R = np.eye(1)
    K, X, eigVals = dlqr(sys.base_model.A.weight.detach().numpy(),
                         sys.base_model.B.weight.detach().numpy(),
                         Q, R)
    for _ in range(100):
        x_init = 15 * np.random.rand(2, 1) - 7.5
        x = x_init
        u = -K @ x

        # Append the initial state to the initial states array
        initial_states.append(torch.from_numpy(x_init.T).reshape((1, 2)))

        # Initialize output and input tensors for the simulation as empty tensors
        output_tensor = torch.empty((0, 2), dtype=torch.float64)
        input_tensor = torch.empty((0, 1), dtype=torch.float64)
        
        while np.linalg.norm(x) > 1e-1:
            u = -K @ x
            x = sys.base_model.A.weight.detach().numpy() @ x + sys.base_model.B.weight.detach().numpy() @ u
            
            # Append the state and input to the input and output arrays
            input_tensor = torch.cat((input_tensor, torch.from_numpy(u).reshape((1, 1))), dim=0)
            output_tensor = torch.cat((output_tensor, torch.from_numpy(x.T).reshape((1, 2))), dim=0)
        
        inputs_arr.append(input_tensor.detach())
        outputs_arr.append(output_tensor.detach())

    # Plot all the simulated trajectories
    plt.figure()
    for i in range(len(inputs_arr)):
        # use matplotlib to plot x-y trajectories
        plt.plot(outputs_arr[i][:, 0], outputs_arr[i][:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simulated trajectories')
    plt.show()

    # Create another system object to learn the model
    learned_sys = System(num_inputs=1, num_states=2)

    # # Initialize the model matrices with the true model matrices
    # learned_sys.set_linear_model_matrices(A=sys.base_model.A.weight.detach(),
    #                                         B=sys.base_model.B.weight.detach())
    
    learned_sys.parse_config(config)
    
    learned_sys.learn_grad(inputs=inputs_arr,
                           true_outputs=outputs_arr,
                           initial_state=initial_states,
                           epochs=100, use_delay=False, use_base_model=True, use_error_model=False)

    # Compare the learned model with the true model
    print('True A: ', sys.base_model.A.weight)
    print('Learned A: ', learned_sys.base_model.A.weight)
    print('True B: ', sys.base_model.B.weight)
    print('Learned B: ', learned_sys.base_model.B.weight)

    # Plot simulation results of first 25 bags of data
    fig, axs = plt.subplots(5,5)
    for bag_idx, ax  in enumerate(axs.ravel()):
        # Plot the simulation of the learned model
        input_arr =inputs_arr[bag_idx]
        output_arr = outputs_arr[bag_idx]
        initial_state = initial_states[bag_idx]

        
        learned_sys.plot_simulation(input_array=input_arr,
                                    true_state=output_arr,
                                    initial_state=initial_state,
                                    ax=ax,
                                    use_delay=False, use_base_model=True, use_error_model=False)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
