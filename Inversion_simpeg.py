import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from discretize import TensorMesh
from SimPEG import maps
from SimPEG.potential_fields import magnetics
from SimPEG import data_misfit, regularization, optimization, inversion, directives, inverse_problem
from SimPEG.utils import mkvc
from SimPEG import data as DataModule
from SimPEG.inversion import BaseInversion, DirectiveList
from SimPEG.directives import BetaEstimate_ByEig, InversionDirective
from SimPEG import utils
from discretize.utils import active_from_xyz
from mpl_toolkits.mplot3d import Axes3D

# Path to the file
file_path = 'Aph23a.xyz'

# Loading the data
data = pd.read_csv(file_path, sep=',\s+', engine='python', header=None)
data.columns = ['X_coordinate', 'Y_coordinate', 'Magnetic_field']
data_x_coordinates = data.iloc[:, 0].to_numpy()
data_y_coordinates = data.iloc[:, 1].to_numpy()  # Now we also have Y-coordinate data
observed_magnetic_field = data.iloc[:, 2].to_numpy()
# Step 1: Data Preparation

# Create a regular grid for the inversion process
# Determine the extents of the survey area
x_min, x_max = data['X_coordinate'].min(), data['X_coordinate'].max()
y_min, y_max = data['Y_coordinate'].min(), data['Y_coordinate'].max()

# Define the number of grid points (can be adjusted based on the desired resolution)
num_points_x = 100  # Example value
num_points_y = 100  # Example value

# Generate a regular grid
grid_x, grid_y = np.linspace(x_min, x_max, num_points_x), np.linspace(y_min, y_max, num_points_y)
grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

# Interpolate the magnetic field data onto the regular grid
grid_magnetic_field = griddata((data['X_coordinate'], data['Y_coordinate']),
                               data['Magnetic_field'], (grid_X, grid_Y), method='cubic')

# Plotting the interpolated magnetic field data
plt.figure(figsize=(10, 8))
plt.contourf(grid_X, grid_Y, grid_magnetic_field, cmap='viridis', levels=100)
plt.colorbar(label='Magnetic Field (nT)')
plt.title('Interpolated Magnetic Field Data')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.show()

# Assuming the depth range (in meters)
depth_min = 0.5  # minimum depth
depth_max = 2.0  # maximum depth

# Extracting min and max values from the data
min_x, max_x = data['X_coordinate'].min(), data['X_coordinate'].max()
min_y, max_y = data['Y_coordinate'].min(), data['Y_coordinate'].max()

# Define the vertical extent of the model (depth)
min_z = -depth_max  # Negative as we're going below the surface
max_z = -depth_min  # Negative as we're going below the surface

# Define the number of cells in each direction (X, Y, Z)
# Adjust these numbers based on the resolution you need
nx, ny, nz = 40, 40, 15

# Define the size of each cell (in meters)
dx, dy, dz = (max_x - min_x) / nx, (max_y - min_y) / ny, (depth_max - depth_min) / nz

# Create a 3D tensor mesh
hx = np.ones(nx) * dx
hy = np.ones(ny) * dy
hz = np.ones(nz) * dz
mesh = TensorMesh([hx, hy, hz], x0="CCN")  # Centered mesh

# Create grid of points for topography
# Lets create a simple Gaussian topo and set the active cells
[xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
b = 100
A = 50
zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))
topo = np.c_[utils.mkvc(xx), utils.mkvc(yy), utils.mkvc(zz)]
# Define an active cells from topo
actv = active_from_xyz(mesh, topo)
nC = int(actv.sum())
# Create a model space (susceptibility) with initial guesses
model = np.ones(3 * nC) * 1e-4

# Mapping from model space to active cells (if you have air cells or inactive cells)
active_cells = mesh.gridCC[:, 2] < 0  # Example: cells below the ground
mapping = maps.IdentityMap(nP=int(active_cells.sum()))

# Assuming 'grid_magnetic_field' is the gridded magnetic field data
observed_data = mkvc(grid_magnetic_field)

# Define uncertainties for the observed data
# This can be a constant value or vary based on data quality
uncertainties = np.full(observed_data.shape, 1)  # Example: all uncertainties set to 1 nT

# Check Mesh Dimensions
total_cells_in_mesh = mesh.nC

# Check Active Cells
number_of_active_cells = active_cells.sum()

# Redefine the starting model to match the number of active cells
starting_model = np.ones(number_of_active_cells) * 1e-4
# Check Active Cells and Starting Model Size
print("Number of active cells:", number_of_active_cells)
print("Size of starting model:", len(starting_model))

# Define the magnetic survey (assuming 'locations' are your observation locations)
rxLoc = np.c_[mkvc(grid_X), mkvc(grid_Y), np.zeros_like(mkvc(grid_X))]
rxList = magnetics.Point(rxLoc)
srcField = magnetics.UniformBackgroundField(receiver_list=[rxList])
survey = magnetics.Survey(srcField)
inducing_field = (48877, 65.1, 3.8)  # Earth's magnetic field (nT): strength, inclination, declination
# Step 3: Inversion Setup
# Define the Forward Model
problem = magnetics.Simulation3DIntegral(
    mesh, survey=survey, chiMap=mapping, ind_active=active_cells
)
problem.survey = survey  # Assuming 'survey' is your observation setup
# Set the inducing field strength
problem.inducingField = inducing_field

# Data Misfit
data_object = DataModule.Data(survey, dobs=observed_data, noise_floor=uncertainties)
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=problem)

# Regularization with reference model set
reg = regularization.WeightedLeastSquares(mesh, active_cells=active_cells, mapping=mapping)
reg.alpha_s = 1e-3  # Smoothness weight
reg.alpha_x = reg.alpha_y = reg.alpha_z = 1  # Anisotropic weights
#reg.reference_model = starting_model
if len(starting_model) == number_of_active_cells:
    reg.reference_model = starting_model
else:
    print("Mismatch in starting model size and number of active cells.")

# Optimization
opt = optimization.ProjectedGNCG(maxIter=20, lower=0., upper=1.)  # Adjust as needed
opt.maxIterLS = 20
opt.maxIterCG = 10
opt.tolCG = 1e-4


# Options for outputting recovered models and predicted data as a dictionary
save_dictionary = directives.SaveOutputEveryIteration(save_txt=False)

irls = directives.Update_IRLS(
f_min_change=1e-4,
max_irls_iterations=20,
minGNiter=1,
beta_tol=0.5,
coolingRate=1,
coolEps_q=True,
sphericalDomain=True,
)

sensitivity_weights = directives.UpdateSensitivityWeights()
update_Jacobi = directives.UpdatePreconditioner()
# Directives for inversion
beta_estimate = BetaEstimate_ByEig()
# Inversion Problem
invProb = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Reorder directive list
directiveList = [
    sensitivity_weights,
    beta_estimate,
    save_dictionary,
    irls,
    update_Jacobi
]

#inv = inversion.BaseInversion(invProb, directiveList=directiveList)

# Run Inversion
#recovered_model = inv.run(starting_model)

# Try running the inversion with a basic setup
try:
    inv = inversion.BaseInversion(invProb, directiveList=[sensitivity_weights, beta_estimate])
    recovered_model = inv.run(starting_model)
except Exception as e:
    print("Error during simplified inversion:", e)


# Reshaping the model to 3D for visualization
recovered_model_reshaped = recovered_model.reshape((nz, ny, nx), order="F")


# Plotting slices at different depths
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Indices for different depths - top, middle, bottom
depth_indices = [0, nz // 2, nz - 1]

for ax, depth_idx in zip(axes, depth_indices):
    # Extracting the slice data for the given depth
    slice_data = recovered_model_reshaped[depth_idx, :, :]

    # Plotting the slice
    contour_plot = ax.contourf(slice_data, cmap="viridis", levels=100)
    fig.colorbar(contour_plot, ax=ax, orientation='vertical')

    ax.set_title(f"Depth slice at index {depth_idx}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

plt.tight_layout()
plt.show()
#x_range = range(10, 15)  # X-axis range for the anomaly (from 10 to 15 inclusive)
#y_range = range(40)  # Y-axis range (entire length)
x_range = range(11, 14)
y_range = range(0, 40)
identified_anomalies = [(x, y) for x in x_range for y in y_range]
# Depth Estimation and Visualization

for anomaly_location in identified_anomalies:  # Replace 'identified_anomalies' with your list of anomaly coordinates
    depth_profile = recovered_model_reshaped[:, anomaly_location[1], anomaly_location[0]]

    plt.figure(figsize=(6, 4))
    plt.plot(depth_profile, range(nz), label='Susceptibility Profile')
    plt.gca().invert_yaxis()  # Invert y-axis to show depth correctly
    plt.xlabel("Susceptibility")
    plt.ylabel("Depth Index")
    plt.title(f"Depth Profile at Location {anomaly_location}")
    plt.legend()
    plt.savefig("/home/dragomir/Downloads/Paleomagnetism/programming in python/munmagtools-master/playground/Thesis_works/Data_processing_and_Inversion/SimPeg/Depth_profiles/" + f"Depth Profile at Location {anomaly_location}" + ".png")
    #plt.show()

# 3D Plotting of the Anomaly
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x_range_3d = range(11, 14)
y_range_3d = range(0, 40)

# Coordinates for the anomaly
x_coords = np.array(x_range_3d) * dx  # Convert grid index to actual coordinates
y_coords = np.array(y_range_3d) * dy

# Extract the model data for the anomaly region
Z = recovered_model_reshaped[:, y_range_3d[0]:y_range_3d[-1]+1, x_range_3d[0]:x_range_3d[-1]+1]

# Reshape Z for plotting
Z_reshaped = Z.reshape(nz, len(y_range_3d), len(x_range_3d))

# Iterate through each depth, y, and x coordinate to plot
for k in range(nz):
    for i in range(len(y_range_3d)):
        for j in range(len(x_range_3d)):
            # Susceptibility value
            susceptibility = Z_reshaped[k, i, j]
            # Plotting the point
            ax.scatter(x_coords[j], y_coords[i], k, c=[susceptibility], cmap='viridis', vmin=Z.min(), vmax=Z.max())

# Set labels and title
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Depth Index')
ax.set_title('3D Visualization of Anomaly Susceptibility')

# Adding a color bar to represent the susceptibility scale
mappable = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=Z.min(), vmax=Z.max()))
mappable.set_array([])
fig.colorbar(mappable, ax=ax, orientation='vertical', label='Susceptibility')

#plt.show()


