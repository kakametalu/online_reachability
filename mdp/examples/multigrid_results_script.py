import pickle
import numpy as np

results = pickle.load(open('multigrid_40.pkl','rb'))


print("Results for fine grid equal to 41 nodes:\n")

t_coarse = results['t_coarse']
t_fine_warm = results['t_fine_warm']
t_fine = results['t_fine']

print("Time for coarse: {}".format(t_coarse))
print("Time for fine w/ warm start: {}".format(t_fine_warm))
print("Time for multigrid: {}".format(t_fine_warm + t_coarse))
print("Time for fine: {}".format(t_fine))

print("\n")

results = pickle.load(open('multigrid_80.pkl','rb'))


print("Results for fine grid equal to 81 nodes:\n")

t_coarse = results['t_coarse']
t_fine_warm = results['t_fine_warm']
t_fine = results['t_fine']

print("Time for coarse: {}".format(t_coarse))
print("Time for fine w/ warm start: {}".format(t_fine_warm))
print("Time for multigrid: {}".format(t_fine_warm + t_coarse))
print("Time for fine: {}".format(t_fine))
