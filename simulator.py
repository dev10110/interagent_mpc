from re import T
import numpy as np
import copy
import interagent_obstacle_mpc as _mpc
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

class Simulator():
  def __init__(self, mpc, x0, xT, dt, T_end = 5.0):
    self.mpc = mpc
    self.dt = dt # simulation timestep
    self.x0 = x0
    self.x = x0
    self.xT = xT # for now, constant target states
    self.t = 0.0
    self.T_end = T_end # when to quit the simulation
    self.paths = None
    self.controls = None

    self.A = np.kron(np.eye(2),np.array([[1, dt], [0, 1]]))
    self.B = np.kron(np.eye(2), np.array([[0.5*dt**2], [dt]]))


  def update_xT(self):
    return

  def simulate(self):

    paths = [[self.x0[i]]  for i in range(self.mpc.num_agents)]
    controls = [[]  for i in range(self.mpc.num_agents)]

    last_update_MPC = self.t
    suc, u_MPC = self.mpc.solve(self.x0, self.xT)
    print(suc)

    for self.t in tqdm(np.arange(start=0,stop=self.T_end, step=self.dt)):
      # print(self.t)
      # get new targets
      self.update_xT()

      # if time to update MPC:
      if (self.t - last_update_MPC) >= self.mpc.T:
        # print("Updating MPC")
        # update MPC
        suc, u_MPC = self.mpc.solve(self.x, self.xT)
        last_update_MPC = self.t
        # print(suc)


      
      # compute the low-level control inputs
      u = copy.deepcopy(u_MPC)
      

      # update dynamics
      for i in range(self.mpc.num_agents):
        self.x[i] = self.A @ self.x[i] + self.B @ u[i]
      
      # save data
      for i in range(self.mpc.num_agents):
        paths[i].append(self.x[i])
        controls[i].append(u[i])

      # increment t
      # self.t += self.dt

    # export data
    self.paths = paths
    self.controls = controls
    
    return paths, controls

  def plot(self):
    if self.paths is None:
      print("Simulate first")
      return

    plt.figure()
    for i in range(self.mpc.num_agents):
      xs = [x[0] for x in self.paths[i]]
      ys = [x[2] for x in self.paths[i]]

      plt.plot(xs, ys, '.-', label=f"Agent {i}")

    for obstacle in self.mpc.obstacles:
      obstacle.plot()
    
    plt.savefig("paths.png")



obstacles = [_mpc.RectObstacles(0.4, 0.4, 0.6, 0.6)]


mpc = _mpc.MPC(number_of_agents=2, obstacles=obstacles, prediction_horizon=20)

x0 = [np.array([0,0,0,0]), np.array([1,0,0,0])]
xT = [np.array([1,0,1,0]), np.array([0,0,1,0])]


sim = Simulator(mpc, x0, xT, dt = 0.01, T_end = 2.0)

sim.simulate()

sim.plot()
    




