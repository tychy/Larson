import matplotlib.pyplot as plt
import numpy as np

# Create the vectors X and Y
x = np.linspace(0.9, 5, 100)
y = 1/x/x/x * (1 - 1/x)
# Create the plot
plt.plot(x,y)

# Add X and y Label
plt.xlabel('x axis')
plt.ylabel('y axis')

# Add a grid
plt.grid(alpha=.4,linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.savefig("spherical.png")
