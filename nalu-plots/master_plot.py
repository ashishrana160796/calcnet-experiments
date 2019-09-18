from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

# defining figure size and dpi for image quality.
fig = plt.figure(figsize=(8, 6), dpi=108)

#================
#  First subplot.
#================

# Set up the axes for the first plot.
ax = fig.add_subplot(2, 2, 1, projection='3d')

phi = 1.61803399

# Two functions for multiplication and creating transformation matrix W.
sigmoid = lambda y: 1 / (1 + pow(phi, -y))
tanh = lambda x: ((pow(phi, x) - pow(phi, -x))/(pow(phi, x)+pow(phi, -x)))


# Generate data for plotting.
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
# Final function for Transformation Matrix W.
Z = tanh(X)*sigmoid(Y)


# Title labelling ,axis labelling and axis rotation.
ax.set_xlabel('tanh(x)', fontsize=14)
ax.set_ylabel('sigmoid(y)', fontsize=14)
ax.set_zlabel('W', fontsize=14)
ax.view_init(30,-130)
ax.set_title('G-NAC')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-2.0, 2.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


#=================
#  Second subplot.
#=================

# Set up the axes for the first plot.
ax = fig.add_subplot(2, 2, 2, projection='3d')

e = 2.732

# Two functions for multiplication and creating transformation matrix W.
sigmoid = lambda y: 1 / (1 + pow(phi, -y))
tanh = lambda x: ((pow(e, x) - pow(e, -x))/(pow(e, x)+pow(e, -x)))


# Generate data for plotting.
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)
# Final function for Transformation Matrix W.
Z = tanh(X)*sigmoid(Y)


# Title labelling ,axis labelling and axis rotation.
ax.set_xlabel('tanh(x)', fontsize=14)
ax.set_ylabel('sigmoid(y)', fontsize=14)
ax.set_zlabel('W', fontsize=14)
ax.view_init(37,-135)
ax.set_title('NAC')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-2.0, 2.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.view_init(37,-135)

#===============
# Third subplot
#===============

# Set up the axes for the first plot.
ax = fig.add_subplot(2, 2, 3, projection='3d')

e = 1.618

# Sub-function decleration for NALU units.
sigmoid = lambda y: 1 / (1 + pow(phi, -y))
tanh = lambda x: ((pow(phi, x)-pow(phi, -x))/(pow(phi, x)+pow(phi, -x)))
# Learned gate decleration.
g_x = lambda x: 1 / (1 + pow(phi, -x))
# Small epsilon value for truly defined exponential space.
eps = 10e-2
# log-exponential space defiined.
m_y = lambda y: pow(phi, np.log(np.abs(y)+eps))

# Generate data for plotting.
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y)

# Final function for Transformation Matrix W.
W_nac = tanh(X)*sigmoid(Y)
# Final function for G-NALU.
Z = ( (g_x(Y)*W_nac) + ((1-g_x(Y))*m_y(Y)) )


# Title labelling ,axis labelling and axis rotation.
ax.set_xlabel('tanh(x)', fontsize=12)
ax.set_ylabel('sig-M/G(y), exp(log(|y|+eps))', fontsize=12)
ax.set_zlabel('Y', fontsize=14)
ax.set_title('G-NALU')
ax.view_init(32,145)


# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-2.0, 2.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


#===============
# Fourth subplot
#===============

# Set up the axes for the first plot.
ax = fig.add_subplot(2, 2, 4, projection='3d')

e = 2.732

# Sub-function decleration for NALU units.
sigmoid = lambda y: 1 / (1 + np.exp(-y))
tanh = lambda x: ((np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x)))
# Learned gate decleration.
g_x = lambda x: 1 / (1 + np.exp(-x))
# Small epsilon value for truly defined exponential space.
eps = 10e-2
# log-exponential space defiined.
m_y = lambda y: np.exp(np.log(np.abs(y)+eps))

# Generate data for plotting.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)

# Final function for Transformation Matrix W.
W_nac = tanh(X)*sigmoid(Y)
# Final function for NALU.
Z = ( (g_x(Y)*W_nac) + ((1-g_x(Y))*m_y(Y)) )


# Title labelling ,axis labelling and axis rotation.
ax.set_xlabel('tanh(x)', fontsize=12)
ax.set_ylabel('sig-M/G(y), exp(log(|y|+eps))', fontsize=12)
ax.set_zlabel('Y', fontsize=14)
ax.set_title('NALU')
ax.view_init(32,145)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-2.0, 2.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


####################
# final plot display
####################
plt.show()
