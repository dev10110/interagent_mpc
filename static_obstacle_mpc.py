import numpy as np
import cvxpy as cp

import matplotlib.pyplot as plt
from matplotlib import patches

# double integrator
# state is [x, vx, y, vy]


T = 0.05 # timestep in seconds
A = np.kron(np.eye(2),np.array([[1, T], [0, 1]]))
B = np.kron(np.eye(2), np.array([[0.5*T**2], [T]]))

class RectObstacles:
  def __init__(self, x_low, y_low, x_high, y_high, margin = 0.05, big_M=1e2):
    self.x_low = x_low
    self.x_high = x_high
    self.y_low = y_low
    self.y_high = y_high
    self.big_M = big_M
    self.margin = margin

  def plot(self):
    
    width = self.x_high - self.x_low
    height = self.y_high - self.y_low

    rect = patches.Rectangle((self.x_low, self.y_low), width, height, edgecolor='black',facecolor='none', lw=2)
    rect_margin = patches.Rectangle((self.x_low-self.margin, self.y_low-self.margin), width+2*self.margin, height+2*self.margin, edgecolor='gray',facecolor='none', lw=1)
    
    plt.gca().add_patch(rect)
    plt.gca().add_patch(rect_margin)
    return

  def create_constraints(self, state):
    binary = cp.Variable(4, boolean=True)
    x = state[0]
    y = state[2]

    constraints = [
      x <= self.x_low - self.margin + binary[0] * self.big_M,
      x >= self.x_high  + self.margin - binary[1] * self.big_M,
      y <= self.y_low  - self.margin + binary[2] * self.big_M,
      y >= self.y_high + self.margin   - binary[3] * self.big_M,
      sum(binary) <= 3
    ]

    return constraints







class MPC:
  def __init__(self, prediction_horizon = 10, number_of_agents = 1, umax=100.0, obstacles=[]):
    
    self.N = prediction_horizon
    self.num_agents = number_of_agents
    self.obstacles = obstacles
    self.umax = umax


    self.solution_X = None
    self.solution_U = None # updated once solve is called

    return

  def plot(self):
    if self.solution_X is None:
      print("Please call solve first")
      return
    
    # plot the trajectories
    plt.figure()
    # plot obstacles
    for obstacle in self.obstacles:
      obstacle.plot()
    
    for i in range(self.num_agents):
      xs = [x[0] for x in self.solution_X[i]]
      ys = [x[2] for x in self.solution_X[i]]
      plt.plot(xs, ys, ".-",label=f"Agent {i}")
      plt.scatter([xs[0]], [ys[0]], marker="o", label=None)
      plt.scatter([xs[-1]], [ys[-1]], marker="x", label=None)

    # plt.grid()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    
    # plt.show()
    plt.savefig("plot.png")
  

  def solve(self, x0, xT):

    # x0 is a list of initial conditions
    # xT is a list of target states

    ### constuct the MPC problem

    X = [[cp.Variable(4) for _ in range(self.N)]  for i in range(self.num_agents)]
    # X[i][k] is the i-th agents state at time k
    U = [[cp.Variable(2) for _ in range(self.N-1)] for i in range(self.num_agents)]

    ### create constraints
    constraints = []

    # initial and final conditions
    for i in range(self.num_agents):
      constraints.append(X[i][0] == x0[i])
      constraints.append(X[i][-1] == xT[i])

    # dynamics constraints
    for i in range(self.num_agents):
      for k in range(self.N-1): # at each timestep
        constraints.append(X[i][k+1] == A @ X[i][k] + B @ U[i][k])
    
    # input constraints
    for i in range(self.num_agents):
      for k in range(self.N-1): 
        constraints.append(cp.norm(U[i][k], "inf") <= self.umax)

    # collision constraints
    for obstacle in self.obstacles:
      for k in range(self.N):
        for i in range(self.num_agents):
          constraints.extend(obstacle.create_constraints(X[i][k]))

    ### construct the objective function
    objective = sum(sum(cp.sum_squares(uk) for uk in Ui) for Ui in U)

    
    ###  call a solver
    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve()

    ### save the trajectory
    self.solution_X = [[x.value for x in Xi] for Xi in X]
    self.solution_U = [[u.value for u in Ui] for Ui in U]

    ### return instantaneous control input
    return [u[0] for u in self.solution_U]


obstacles = [RectObstacles(0.4, 0.4, 0.6, 0.6)]


mpc = MPC(number_of_agents=2, obstacles=obstacles, prediction_horizon=20)

x0 = [np.array([0,0,0,0]), np.array([1,0,0,0])]
xT = [np.array([1,0,1,0]), np.array([0,0,1,0])]
mpc.solve(x0, xT)

mpc.plot()