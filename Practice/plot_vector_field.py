# Import required modules
import numpy as np
import matplotlib.pyplot as plt

# Meshgrid
x, y = np.meshgrid(np.linspace(-3, 3, 30),
                   np.linspace(-3, 3, 30))

# Directional vectors
xdot = np.sin(x)
ydot = np.cos(y)

# Plotting Vector Field with QUIVER
plt.quiver(x, y, xdot, ydot, color='g')
plt.title('Vector Field')

# Setting x, y boundary limits
plt.xlim(-4, 4)
plt.ylim(-4, 4)

# Show plot with grid
plt.grid()
plt.show()