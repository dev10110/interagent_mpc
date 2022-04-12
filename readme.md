# Interagent MPC

Generates trajectories for multiple agents to avoid static obstacles and each other

The mpc problem is a mixed integer QP, solved using default methods in CVXPY. 

## usage  
clone the repo, run
`python3 simulator.py`


useful references:
```
@INPROCEEDINGS{1024509,
  author={Bellingham, J. and Richards, A. and How, J.P.},
  booktitle={Proceedings of the 2002 American Control Conference (IEEE Cat. No.CH37301)}, 
  title={Receding horizon control of autonomous aerial vehicles}, 
  year={2002},
  volume={5},
  number={},
  pages={3741-3746 vol.5},
  doi={10.1109/ACC.2002.1024509}}
```

```
@article{KON2009529,
title = {Trajectory Generation based on Model Predictive Control with Obstacle Avoidance between Prediction Time Steps},
journal = {IFAC Proceedings Volumes},
volume = {42},
number = {16},
pages = {529-535},
year = {2009},
note = {9th IFAC Symposium on Robot Control},
issn = {1474-6670},
doi = {https://doi.org/10.3182/20090909-4-JP-2010.00090},
url = {https://www.sciencedirect.com/science/article/pii/S147466701530690X},
author = {Kazuyuki Kon and Hiroaki Fukushima and Fumitoshi Matsuno},
}
```