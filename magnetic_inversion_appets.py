import matplotlib
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

from discretize import TensorMesh
from discretize.utils import active_from_xyz
from SimPEG.potential_fields import magnetics
from SimPEG import dask
from SimPEG.utils import plot2Ddata, surface2ind_topo
from SimPEG import (
    maps,
    data,
    inverse_problem,
    data_misfit,
    regularization,
    optimization,
    directives,
    inversion,
    utils,
)
import pyvista as pv
import pandas as pd
from ipywidgets import widgets, interact


def read_ubc_magnetic_data(data_filename):
    data = pd.read_csv(data_filename, sep=',\s+', engine='python', header=None)
    z_column = []
    # Renaming columns for clarity
    data.columns = ['x', 'y', 'data']
    for i in range(len(data['x'])):
        z_column.append(2.0)

    data.insert(3, "z", z_column, True)

    meta_data = {}
    meta_data['inclination'] = 64.37
    meta_data['declination'] = 4.15
    meta_data['b0'] = 48722.8

    return data, meta_data

data_filename = "Aph23a.xyz"
df, meta_data = read_ubc_magnetic_data(data_filename)

# Down sample the data
matplotlib.rcParams['font.size'] = 14
nskip = 2
receiver_locations = df[['x', 'y', 'z']].values #[::nskip,:]
dobs = df['data'].values #[::nskip]
# Plot
fig = plt.figure(figsize=(12, 10))
vmin, vmax = np.percentile(dobs, 0.5), np.percentile(dobs, 99.5)
tmp = np.clip(dobs, vmin, vmax)
ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
plot2Ddata(
    receiver_locations,
    tmp,
    ax=ax1,
    ncontour=30,
    clim=(vmin-5, vmax+5),
    contourOpts={"cmap": "Spectral_r"},
)
ax1.set_title("TMI Anomaly")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")

ax2 = fig.add_axes([0.9, 0.25, 0.05, 0.5])

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.Spectral_r
)
cbar.set_label("$nT$", rotation=270, labelpad=15, size=12)

plt.show()

maximum_anomaly = np.max(np.abs(dobs))
standard_deviation = 0.02 * maximum_anomaly * np.ones(len(dobs)) + 2
#standard_deviation = 0.02 * abs(dobs) + 2

# Define the component(s) of the field we are inverting as a list. Here we will
# Invert total magnetic intensity data.
components = ["tmi"]

# Use the observation locations and components to define the receivers. To
# simulate data, the receivers must be defined as a list.
receiver_list = magnetics.receivers.Point(receiver_locations, components=components)

receiver_list = [receiver_list]

# Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
inclination = meta_data['inclination']
declination = meta_data['declination']
strength = meta_data['b0']
inducing_field = (strength, inclination, declination)

source_field = magnetics.sources.UniformBackgroundField(
    receiver_list=receiver_list, parameters=inducing_field
)

# Define the survey
survey = magnetics.survey.Survey(source_field)

data_object = data.Data(survey, dobs=dobs, standard_deviation=standard_deviation)

mesh = TensorMesh.read_UBC('./data/Raglan_1997/mesh_appets.msh')
susceptibility_ubc = mesh.read_model_UBC('./data/Raglan_1997/maginv3d.sus')

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
mesh.plot_slice(susceptibility_ubc*np.nan, ax=ax, grid=True)
ax.plot(receiver_locations[:,0], receiver_locations[:,1], 'r.')
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
ax.set_aspect(1)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
mesh.plot_slice(susceptibility_ubc*np.nan, ax=ax, grid=True, normal='Y')
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Depth (m)")
ax.set_aspect(1)

# Define background susceptibility model in SI. Don't make this 0!
# Otherwise, the gradient for the 1st iteration is zero and the inversion will
# not converge.
background_susceptibility = 1e-4

# Find the indecies of the active cells in forward model (ones below surface)
ind_active = active_from_xyz(mesh, np.c_[receiver_locations[:,:2], np.zeros(survey.nD)])

# Define mapping from model to active cells
nC = int(ind_active.sum())
model_map = maps.IdentityMap(nP=nC)  # model consists of a value for each cell

# Define starting model
starting_model = background_susceptibility * np.ones(nC)
reference_model = np.zeros(nC)

simulation = magnetics.Simulation3DIntegral(
    mesh=mesh,
    survey=survey,
    #chi=starting_model,  # you need to define or calculate this
    chiMap=model_map,
    model_type="scalar",  # Use the appropriate model_type
    #is_amplitude_data=False,
    ind_active=ind_active# Set this based on your data type
    # include other necessary keyword arguments if needed
)

# Define the data misfit. Here the data misfit is the L2 norm of the weighted
# residual between the observed data and the data predicted for a given model.
# Within the data misfit, the residual between predicted and observed data are
# normalized by the data's standard deviation.
dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)

# Define the regularization (model objective function)
reg = regularization.Sparse(
    mesh,
    indActive=ind_active,
    mapping=model_map,
    alpha_s=1,
    alpha_x=1,
    alpha_y=1,
    alpha_z=1,
)

# Define sparse and blocky norms p, qx, qy, qz
reg.norms = [0, 2, 2, 2]

# Define how the optimization problem is solved. Here we will use a projected
# Gauss-Newton approach that employs the conjugate gradient solver.
opt = optimization.ProjectedGNCG(
    maxIter=15, lower=0.0, upper=np.Inf, maxIterLS=20, maxIterCG=30, tolCG=1e-3
)

# Here we define the inverse problem that is to be solved
inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

# Defining a starting value for the trade-off parameter (beta) between the data
# misfit and the regularization.
starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=5)

# the cooling schedule for the trade-off parameter.
#beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=1)

# Defines the directives for the IRLS regularization. This includes setting
# the cooling schedule for the trade-off parameter.
update_IRLS = directives.Update_IRLS(
    f_min_change=1e-4,
    max_irls_iterations=30,
    coolEpsFact=1.5,
    beta_tol=1e-2,
)

# Options for outputting recovered models and predicted data as a dictionary
save_dictionary = directives.SaveOutputEveryIteration(save_txt=True)

# Updating the preconditionner if it is model dependent.
update_jacobi = directives.UpdatePreconditioner()

# Setting a stopping criteria for the inversion.
target_misfit = directives.TargetMisfit(chifact=1)

# Add sensitivity weights
sensitivity_weights = directives.UpdateSensitivityWeights(every_iteration=False)
opt.remember('xc')
# The directives are defined as a list.
directives_list = [
    sensitivity_weights,
    starting_beta,
    save_dictionary,
    #beta_schedule,
    update_IRLS,
    update_jacobi,
    target_misfit
]

inv = inversion.BaseInversion(inv_prob, directiveList=directives_list)
recovered_model = inv.run(starting_model)

dpred = simulation.dpred(m=recovered_model)
fname = "magnetics_data.obs"
fily = np.savetxt(fname, np.c_[receiver_locations, dpred], fmt="%.4e")

# Plot Recovered Model
fig = plt.figure(figsize=(6, 5.5))

ax1 = fig.add_axes([0.15, 0.15, 0.65, 0.75])
mesh.plot_image(recovered_model, ax=ax1, grid=True, pcolor_opts={"cmap": "viridis"})
ax1.set_title("Recovered Model")

ax2 = fig.add_axes([0.82, 0.15, 0.05, 0.75])
norm = mpl.colors.Normalize(vmin=np.min(recovered_model), vmax=np.max(recovered_model))
cbar = mpl.colorbar.ColorbarBase(
    ax2, norm=norm, orientation="vertical", cmap=mpl.cm.viridis
)
cbar.set_label("Susceptibility", rotation=270, labelpad=15, size=12)

plt.show()

# Calculate the difference between the observed and predicted data
data_difference = dobs - dpred

# Plotting
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

# Plot original data
axs[0].scatter(receiver_locations[:, 0], receiver_locations[:, 1], c=dobs, cmap='viridis')
axs[0].set_title('Original Data')
axs[0].set_xlabel('Easting [m]')
axs[0].set_ylabel('Northing [m]')
#axs[0].set_colorbars(label='nT')

# Plot recovered model data
axs[1].scatter(receiver_locations[:, 0], receiver_locations[:, 1], c=dpred, cmap='viridis')
axs[1].set_title('Recovered Model Data')
axs[1].set_xlabel('Easting [m]')
axs[1].set_ylabel('Northing [m]')
#axs[1].set_colorbars(label='nT')

# Plot difference
axs[2].scatter(receiver_locations[:, 0], receiver_locations[:, 1], c=data_difference, cmap='viridis')
axs[2].set_title('Data Difference')
axs[2].set_xlabel('Easting [m]')
axs[2].set_ylabel('Northing [m]')
#axs[2].set_colorbars(label='nT')

plt.tight_layout()
plt.show()

# # First, create a volumetric dataset from the mesh and recovered model
# vol = mesh.to_vtk(cell_data={"susceptibility": recovered_model})
#
# # Then, plot using PyVista's plotter
# plotter = pv.Plotter()
# plotter.add_mesh(vol, cmap="viridis", show_edges=True, scalar_bar_args={"title": "Susceptibility"})
# plotter.view_xy()
# plotter.show()

# Plot original data
plt.scatter(receiver_locations[:, 0], receiver_locations[:, 1], c=dobs, cmap='viridis')
plt.title('Original Data')
plt.xlabel('Easting [m]')
plt.ylabel('Northing [m]')
plt.show()

print("recovered model: ", len(recovered_model))
print("measured data: ", len(df['data']))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

# Assuming 'susceptibility_ubc' is your recovered model, and you're interested in plotting a slice.
# If your recovered model is not on the original mesh scale or needs to be activated only for certain cells, adjust accordingly.

# Plot a horizontal slice at a specific depth or a vertical slice along a specific direction
slice_position = 0  # Adjust this based on your mesh and what you want to show
normal = 'Z'  # Choose 'X', 'Y', or 'Z' for the slice direction

# For plotting a slice, you need to use the plot_slice method of the mesh object
ax = mesh.plot_slice(recovered_model, normal=normal, ax=ax, grid=True, loc=slice_position, pcolor_opts={"cmap": "viridis"})

# ax.set_xlabel("Easting [m]")
# ax.set_ylabel("Northing [m]" if normal == 'Z' else "Depth [m]")
# ax.set_title("Recovered Model Slice")
plt.show()


# Assuming x_point and y_point are your coordinates of interest
x_point, y_point = 12, 17

# Find horizontal cell closest to (x_point, y_point)
dx = np.abs(mesh.cell_centers_x - x_point)
dy = np.abs(mesh.cell_centers_y - y_point)
closest_cell_x = np.argmin(dx)
closest_cell_y = np.argmin(dy)

# Assuming ind_active is a boolean array marking active cells
z_active = mesh.cell_centers_z[ind_active]

# Assuming you have a way to link 3D indices to 1D model indices
# This can be complex without direct access to indices attribute and depends on mesh structure
# For simplicity, let's proceed with plotting assuming we have z coordinates and corresponding model values

# Plotting the profile
plt.figure(figsize=(6, 8))
plt.plot(recovered_model[ind_active], z_active, '-o')  # This line needs adjustment to match your model's indexing
plt.gca().invert_yaxis()
plt.xlabel('Recovered Model Value')
plt.ylabel('Depth [m]')
plt.title('Depth Profile at (x=12, y=17)')
plt.grid(True)
plt.show()

'''
file_path = 'InversionModel-2024-01-17-15-49.txt'

paramters_dictionary = {
    "beta": [],
    "phi_d": [],
    "phi_m": [],
    "f": [],
    "|proj(x-g)-x|": [],
    "LS": [],
    "Comment": []
}


with open(file_path, 'r') as file:
    for line in file:
        # Skip lines starting with '#' or empty lines
        if line.startswith('#') or line.strip() == '':
            continue

        # Split the line into components
        parts = line.split()
        if len(parts) < 7:  # Adjust based on the expected number of columns
            continue  # Skip lines that don't have enough columns

        # Check if the first part is a number
        try:
            float(parts[0])
        except ValueError:
            continue  # Skip the line if the first part is not a number

        # Append data to the dictionary
        paramters_dictionary["beta"].append(float(parts[0]))
        paramters_dictionary["phi_d"].append(float(parts[1]))
        paramters_dictionary["phi_m"].append(float(parts[2]))
        paramters_dictionary["f"].append(float(parts[3]))
        paramters_dictionary["|proj(x-g)-x|"].append(float(parts[4]))
        paramters_dictionary["LS"].append(float(parts[5]))
        paramters_dictionary["Comment"].append(parts[6])

def plot_tikhonov_curve(iteration, scale):
    phi_d = paramters_dictionary['phi_d']
    phi_m = paramters_dictionary['phi_m']
    beta = paramters_dictionary['beta']

    iterations = np.arange(len(phi_d)) + 1  # Number of iterations is based on the length of one of the lists

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    axs[0].plot(phi_m ,phi_d, 'k.-')
    axs[0].plot(phi_m[iteration-1] ,phi_d[iteration-1], 'go', ms=10)
    axs[0].set_xlabel("$\phi_m$")
    axs[0].set_ylabel("$\phi_d$")
    axs[0].grid(True)

    axs[1].plot(iterations, phi_d, 'k.-')
    axs[1].plot(iterations[iteration-1], phi_d[iteration-1], 'go', ms=10)
    ax_1 = axs[1].twinx()
    ax_1.plot(iterations, phi_m, 'r.-')
    ax_1.plot(iterations[iteration-1], phi_m[iteration-1], 'go', ms=10)
    axs[1].set_ylabel("$\phi_d$")
    ax_1.set_ylabel("$\phi_m$")
    axs[1].set_xlabel("Iterations")
    axs[1].grid(True)
    axs[0].set_title(
        "$\phi_d$={:.1e}, $\phi_m$={:.1e}, $\\beta$={:.1e}".format(phi_d[iteration-1], phi_m[iteration-1], beta[iteration-1]),
        fontsize = 14
    )
    axs[1].set_title("Target misfit={:.0f}".format(survey.nD/2))
    for ii, ax in enumerate(axs):
        if ii == 0:
            ax.set_xscale(scale)
        ax.set_yscale(scale)
        xlim = ax.get_xlim()
        ax.hlines(survey.nD/2, xlim[0], xlim[1], linestyle='--', label='$\phi_d^{*}$')
        ax.set_xlim(xlim)
    axs[0].legend()
    plt.tight_layout()

interact(
    plot_tikhonov_curve, iteration=widgets.IntSlider(min=1, max=len(paramters_dictionary), step=1),
    scale=widgets.RadioButtons(options=["linear", "log"])
)

def plot_dobs_vs_dpred(iteration):
    # Predicted data with final recovered model
    #dpred = paramters_dictionary[iteration]['dpred']

    # Observed data | Predicted data | Normalized data misfit
    data_array = np.c_[dobs, dpred, (dobs - dpred) / standard_deviation]
    vmin, vmax = dobs.min(), dobs.max()
    fig = plt.figure(figsize=(17, 4))
    plot_title = ["Observed", "Predicted", "Normalized Misfit"]
    plot_units = ["nT", "nT", ""]

    ax1 = 3 * [None]
    ax2 = 3 * [None]
    norm = 3 * [None]
    cbar = 3 * [None]
    cplot = 3 * [None]
    v_lim = [(vmin, vmax), (vmin, vmax),(-3,3)]

    for ii in range(0, 3):

        ax1[ii] = fig.add_axes([0.33 * ii + 0.03, 0.11, 0.25, 0.84])
        cplot[ii] = plot2Ddata(
            receiver_list[0].locations,
            data_array[:, ii],
            ax=ax1[ii],
            ncontour=30,
            clim=v_lim[ii],
            contourOpts={"cmap": "Spectral_r"},
        )
        ax1[ii].set_title(plot_title[ii])
        ax1[ii].set_xlabel("x (m)")
        ax1[ii].set_ylabel("y (m)")

        ax2[ii] = fig.add_axes([0.33 * ii + 0.27, 0.11, 0.01, 0.84])
        norm[ii] = mpl.colors.Normalize(vmin=v_lim[ii][0], vmax=v_lim[ii][1])
        cbar[ii] = mpl.colorbar.ColorbarBase(
            ax2[ii], norm=norm[ii], orientation="vertical", cmap=mpl.cm.Spectral_r
        )
        cbar[ii].set_label(plot_units[ii], rotation=270, labelpad=15, size=12)
    for ax in ax1[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    plt.show()

interact(plot_dobs_vs_dpred, iteration=widgets.IntSlider(min=1, max=len(paramters_dictionary), step=1, value=1))

def plot_recovered_model(iteration, xslice, yslice, zslice, vmax):
    fig = plt.figure(figsize=(10, 10))
    mesh.plot_3d_slicer(
        recovered_model, clim=(0, vmax),
        xslice=xslice,
        yslice=yslice,
        zslice=zslice,
        fig=fig,
        pcolor_opts={'cmap':'Spectral_r'}
    )
interact(
    plot_recovered_model,
    iteration=widgets.IntSlider(min=1, max=len(paramters_dictionary), value=0),
    xslice=widgets.FloatText(value=2000, step=100),
    yslice=widgets.FloatText(value=41000, step=100),
    zslice=widgets.FloatText(value=-500, step=100),
    vmax=widgets.FloatText(value=0.07),
)
'''

def plot_3d_with_pyvista(model, notebook=True, threshold=0.04):
    pv.set_plot_theme("document")
    pv.global_theme.allow_empty_mesh = True
    # Get the PyVista dataset of the inverted model
    dataset = mesh.to_vtk({'susceptibility':model})
    # Create the rendering scene
    p = pv.Plotter()
    # add a grid axes
    p.show_grid()
    # Extract volumetric threshold
    threshed = dataset.threshold(threshold, invert=False)
    # Add spatially referenced data to the scene
    dparams = dict(
        show_edges=False,
        cmap="Spectral_r",
        clim=[0, 0.0015],
        #scalar_bar_args='Scalar Bar Title',
    )
    p.add_mesh(threshed, **dparams)
    p.set_scale(1,1,1)
    #cpos = [(-5090.61095767987, 35424.20054814459, 5280.45524943451),
     #(2298.051317829793, 40974.692421295964, -864.2486811315523),
     #(0.4274014723619113, 0.35262874486945933, 0.8324547733749025)]
    #p.camera_position = cpos
    p.show(window_size=[100, 100])

plot_3d_with_pyvista(recovered_model, notebook=False, threshold=0.0002)
