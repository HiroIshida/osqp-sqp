import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

with open("/tmp/osqp_sqp_cpp_test-double_integrator.csv") as f:
    reader = csv.reader(f)
    data = list(reader)
data = np.array([float(x[0]) for x in data]).reshape(-1, 6)
X = data[:, :2]

fig, ax = plt.subplots()
ax.add_patch(plt.Polygon([[0, 0], [1, 0], [1, 1], [0, 1]], closed=True, fill=False))
ax.plot(X[:, 0], X[:, 1], "bo-", markersize=1)   
ax.plot(0.1, 0.1, "ko")
ax.plot(0.9, 0.95, "r*", markersize=10)
ax.add_patch(plt.Circle((0.25, 0.4), 0.2, fill=True, color="grey"))
ax.add_patch(plt.Circle((0.75, 0.7), 0.15, fill=True, color="grey"))
plt.show()
