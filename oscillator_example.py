import argparse
from control_analysis_pipeline.system.system import System
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from yaml import load
from yaml import Loader
import time

def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
    
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151 - http://www.mwm.im/lqr-controllers-with-python/
    
    #first, try to solve the ricatti equation
    P = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*P*B+R)*(B.T*P*A))
    
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    return K, P, eigVals

def main():
    config = None
    with open("double_int_config.yaml", "r") as stream:
        config = load(stream, Loader=Loader)

    sys = System()
    sys.parse_config(config)

    # Double integrator system
    sys.set_linear_model_matrices(A=torch.tensor([[0, 0.5], [-0.5, 0]]),
                                  B=torch.tensor([[0], [0.1]]))

    # Simulate the system closed loop with LQR controller (no delay)
    # Simulation ends when the state is within 1e-15 of the origin
    # Repeat the simulation for 100 times to collect a dataset  
    inputs_arr = []
    outputs_arr = []
    initial_states = []
    
    learn_with_controller = False
    with_noise = False
    Q = np.eye(2)
    R = np.eye(1)
    K, X, eigVals = dlqr(sys.base_model.A.weight.detach().numpy(),
                         sys.base_model.B.weight.detach().numpy(),
                         Q, R)
    simulation_horizon = 10
    for j in range(100):
        x_init = 15 * np.random.rand(2, 1) - 7.5
        x = x_init
        if learn_with_controller:
            u = -K @ x
        else:
            # oscillating input to excite the system
            u = np.array([[-1.0 * np.sin(0.1 * 0)]])

        # Append the initial state to the initial states array
        initial_states.append(torch.from_numpy(x_init.T).flatten())

        # Initialize output and input tensors for the simulation as empty tensors
        output_tensor = torch.empty((0, 2))
        input_tensor = torch.empty((0, 1))
        
        for i in range(simulation_horizon):
            if learn_with_controller:
                u = -K @ x
            else:
                # oscillating input to excite the system
                u = np.array([[-1.0 * np.sin(0.1 * (j+i))]])

            # Simulate the system
            x = sys.base_model.A.weight.detach().numpy() @ x + sys.base_model.B.weight.detach().numpy() @ u
            
            if with_noise:
                # Add noise to the state
                xobs = x + 0.1 * np.random.randn(2, 1)
            else:
                xobs = x

            # Append the state and input to the input and output arrays
            input_tensor = torch.cat((input_tensor, torch.from_numpy(u).reshape((1, 1))), dim=0)
            output_tensor = torch.cat((output_tensor, torch.from_numpy(xobs.T).reshape((1, 2))), dim=0)
        
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

    # Plot the distribution of control inputs
    plt.figure()
    for i in range(len(inputs_arr)):
        # use matplotlib to plot x-y trajectories
        plt.plot(inputs_arr[i][:, 0])
    plt.xlabel('Time')
    plt.ylabel('Control input')
    plt.title('Control inputs')
    plt.show()

    # Create another system object to learn the model
    learned_sys = System(num_actions=1, num_states=2)

    # # Initialize the model matrices with the true model matrices
    # learned_sys.set_linear_model_matrices(A=sys.base_model.A.weight.detach(),
    #                                         B=sys.base_model.B.weight.detach())
    
    learned_sys.parse_config(config)
    
    # # Set only B-matrix to the true model matrices
    # learned_sys.set_linear_model_matrices(A=None,B=torch.zeros_like(sys.base_model.B.weight.detach()))

    # # Freeze the B-matrix of the learned model
    # for param in learned_sys.base_model.B.parameters():
    #     param.requires_grad = False
    
    tick = time.time()
    learned_sys.learn_base_grad(inputs=inputs_arr,
                           true_outputs=outputs_arr,
                           initial_state=initial_states,
                           batch_size=10,
                           stop_threshold=1e-5,
                           epochs=100)
    tock = time.time()
    print('Time taken to learn the model: ', tock - tick)

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
    # top right legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.show()

    # Compare the learned model with the true model closed loop with LQR controller (no delay)
    # Simulation ends when the state is within 1e-15 of the origin
    # Simulation is done using x = Ax + Bu and not model.forward()
    inputs_arr = []
    outputs_arr = []
    initial_states = []
    
    Q = np.eye(2)
    R = np.eye(1)
    Kbase, _, _ = dlqr(sys.base_model.A.weight.detach().numpy(),
                         sys.base_model.B.weight.detach().numpy(),
                         Q, R)
    Klearned, _, _ = dlqr(learned_sys.base_model.A.weight.detach().numpy(),
                          learned_sys.base_model.B.weight.detach().numpy(),
                          Q, R)
    for _ in range(1):
        x_init = 15 * np.random.rand(2, 1) - 7.5
        xbase = x_init.copy()
        xlearned = x_init.copy()
        ubase = -Kbase @ x
        ulearned = -Klearned @ x
        
        # Append the initial state twice to the initial states array
        initial_states.append(torch.from_numpy(x_init.T))
        initial_states.append(torch.from_numpy(x_init.T))

        # Initialize output and input tensors for the simulation as empty tensors
        output_tensor = torch.from_numpy(x_init.T).reshape((1, 2))
        input_tensor = torch.empty((0, 1))

        while np.linalg.norm(xbase) > 1e-1:
            ubase = -Kbase @ xbase
            xbase = sys.base_model.A.weight.detach().numpy() @ xbase + sys.base_model.B.weight.detach().numpy() @ ubase

            # Append the state and input to the input and output arrays
            input_tensor = torch.cat((input_tensor, torch.from_numpy(ubase).reshape((1, 1))), dim=0)
            output_tensor = torch.cat((output_tensor, torch.from_numpy(xbase.T).reshape((1, 2))), dim=0)
            
        inputs_arr.append(input_tensor.detach())
        outputs_arr.append(output_tensor.detach())

        # Initialize output and input tensors for the simulation as empty tensors
        output_tensor = torch.from_numpy(x_init.T).reshape((1, 2))
        # Apply inputs from the controller of the base model to the learned model
        for i in range(input_tensor.shape[0]):
            ulearned = input_tensor[i, :].detach().numpy().reshape((1, 1))
            xlearned = learned_sys.base_model.A.weight.detach().numpy() @ xlearned + learned_sys.base_model.B.weight.detach().numpy() @ ulearned

            # Append the state and input to the input and output arrays
            output_tensor = torch.cat((output_tensor, torch.from_numpy(xlearned.T).reshape((1, 2))), dim=0)

        inputs_arr.append(input_tensor.detach())
        outputs_arr.append(output_tensor.detach())

    # Plot all the simulated trajectories
    plt.figure()
    markers = ['o', '.']
    colors = ['b', 'r']
    for i in range(len(inputs_arr)):
        # use matplotlib to plot x-y trajectories
        plt.plot(outputs_arr[i][:, 0], outputs_arr[i][:, 1], marker=markers[i], markersize=10, color=colors[i])
        # Add a circle at the initial state
        plt.plot(initial_states[i][:, 0], initial_states[i][:, 1], marker=markers[i], markersize=10, color=colors[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Simulated trajectories')
    plt.show()

    # Save the learned model
    learned_sys.save_to_json('learned_model.json')
    
if __name__ == "__main__":
    main()
