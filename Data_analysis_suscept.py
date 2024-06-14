import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data_filename = "Aph23geo.csv"
data = pd.read_csv(data_filename, sep=',', engine='python', header=None)

# Now try to access the 'sample ID' column
data.columns = ['sample_ID','sample_depth','empty_container_mass','total_mass','sample_mass','container_volume','sample_density','Kappa_at_0.46kHz_LF_(vol.)','Kappa_at_4.6kHz_HF_(vol.)','Chi-LF_(low-freq.)','Kappa_FD','Kappa_FD%_(freq.-dep.)','Chi_FD']
data['sample_ID'] = ['1_1', '1_2', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '5_1', '5_2', '5_3', '5_4', '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7', '6_8', '6_9', '6_10', '6_11']
data['sample_depth'] = [35, 80, 32, 50, 63, 82, 91, 130, 142, 163, 33, 53, 62, 77, 36, 52, 67, 85, 152, 180, 36, 53, 76, 92, 42, 68, 84, 145, 150, 158, 178, 184, 200, 245, 277]
data['empty_container_mass'] = [4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1, 4.1]
data['total_mass'] = [27.78, 24.75, 25.37, 27.41, 24.47, 24.61, 22.75, 22.42, 24.21, 22.71, 27.49, 25.93, 24.6, 21.49, 23.81, 23.3, 24.47, 23.31, 24.43, 22.66, 28.97, 24.12, 23.44, 26.07, 23.53, 22.0, 21.72, 24.02, 21.26, 22.48, 22.52, 25.72, 23.75, 23.4, 30.29]
data['sample_mass'] = [23.68, 20.65, 21.27, 23.31, 20.37, 20.51, 18.65, 18.32, 20.11, 18.61, 23.39, 21.83, 20.5, 17.39, 19.71, 19.2, 20.37, 19.21, 20.33, 18.56, 24.87, 20.02, 19.34, 21.97, 19.43, 17.9, 17.62, 19.92, 17.16, 18.38, 18.42, 21.62, 19.65, 19.3, 26.19]
data['container_volume'] = [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
data['sample_density'] = [1.973, 1.721, 1.773, 1.943, 1.698, 1.709, 1.554, 1.527, 1.676, 1.551, 1.949, 1.819, 1.708, 1.449, 1.643, 1.6, 1.698, 1.601, 1.694, 1.547, 2.073, 1.668, 1.612, 1.831, 1.619, 1.492, 1.468, 1.66, 1.43, 1.532, 1.535, 1.802, 1.638, 1.608, 2.183]
data['Kappa_at_0.46kHz_LF_(vol.)'] = [768, 67, 892, 1041, 854, 938, 1095, 707, 805, 9, 957, 164, 95, 3, 777, 582, 23, 12, 10, 22, 968, 782, 512, 131, 745, 484, 448, 387, 448, 274, 510, 630, 58, 17, 16]
data['Kappa_at_4.6kHz_HF_(vol.)'] = [698, 59, 802, 935, 763, 956, 964, 624, 709, 13, 853, 158, 67, 4, 683, 510, 92, 13, 14, 3, 873, 698, 456, 126, 651, 435, 392, 349, 416, 253, 462, 579, 21, 18, 21]
data['Chi-LF_(low-freq.)'] = [0.389, 0.039, 0.503, 0.536, 0.503, 0.549, 0.705, 0.463, 0.48, 0.006, 0.491, 0.09, 0.056, 0.002, 0.473, 0.364, 0.014, 0.007, 0.006, 0.014, 0.467, 0.469, 0.318, 0.072, 0.46, 0.324, 0.305, 0.233, 0.313, 0.179, 0.332, 0.35, 0.035, 0.011, 0.007]
data['Kappa_FD'] = [70, 8, 90, 106, 91, -18, 131, 83, 96, -4, 104, 6, 28, -1, 94, 72, -69, -1, -4, 19, 95, 84, 56, 5, 94, 49, 56, 38, 32, 21, 48, 51, 37, -1, -5]
data['Kappa_FD%_(freq.-dep.)'] = [9.115, 11.94, 10.09, 10.183, 10.656, -1.919, 11.963, 11.74, 11.925, -44.444, 10.867, 3.659, 29.474, -33.333, 12.098, 12.371, -300.0, -8.333, -40.0, 86.364, 9.814, 10.742, 10.938, 3.817, 12.617, 10.124, 12.5, 9.819, 7.143, 7.664, 9.412, 8.095, 63.793, -5.882, -31.25]
data['Chi_FD'] = [0.035, 0.005, 0.051, 0.055, 0.054, -0.011, 0.084, 0.054, 0.057, -0.003, 0.053, 0.003, 0.016, -0.001, 0.057, 0.045, -0.041, -0.001, -0.002, 0.012, 0.046, 0.05, 0.035, 0.003, 0.058, 0.033, 0.038, 0.023, 0.022, 0.014, 0.031, 0.028, 0.023, -0.001, -0.002]

# Apply the filters
data = data[data['Kappa_at_0.46kHz_LF_(vol.)'] >= 30]
data = data[data['Kappa_FD'] >= 0]
data = data[(data['Kappa_FD%_(freq.-dep.)'] >= 2) & (data['Kappa_FD%_(freq.-dep.)'] <= 17)]
print(len(data['sample_ID']))
# Sample susceptibilities for each depth layer (replace with your actual data)
susceptibilities = data['Kappa_at_0.46kHz_LF_(vol.)']
depth_list= data['sample_depth']


# Define the color mapping based on the first digit
color_map = {'1': 'red', '2': 'blue', '3': 'black', '4': 'pink', '5': 'orange', '6': 'green'}

# Create a figure and a set of subplots
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Iterate over each plot configuration
for a, x_col, y_col, title, xlabel, ylabel in zip(
    ax,
    ['Kappa_FD', 'Kappa_at_0.46kHz_LF_(vol.)', 'Kappa_at_0.46kHz_LF_(vol.)'],
    ['Kappa_FD%_(freq.-dep.)', 'Kappa_FD', 'Kappa_FD%_(freq.-dep.)'],
    ['', '', ''],#[r'$\kappa_{fd}\%$ vs $\kappa_{fd}$', r'$\kappa_{fd} \times 10^{-6}$ vs $\kappa_{lf} \times 10^{-6}$', r'Trends of $\kappa_{fd}\%$ values vs $\kappa_{lf}$'],
    [r'$\kappa_{fd} \times 10^{-6}$', r'$\kappa_{lf} \times 10^{-6}$', r'$\kappa_{lf} \times 10^{-6}$'],
    ['$\kappa_{fd}\%$', r'$\kappa_{fd} \times 10^{-6}$', '$\kappa_{fd}\%$']
):
    # Scatter plot
    a.scatter(data[x_col], data[y_col], marker='o', color='black')
    # Loop to add each label
    for i, txt in enumerate(data['sample_ID']):
        # Extract the first digit and get the corresponding color
        color = color_map.get(txt[0], 'gray')  # Default to 'gray' if not found
        # Add text with the specific color
        a.text(data[x_col].iloc[i], data[y_col].iloc[i], txt, fontsize=12, ha='left', va='bottom', color=color)
    # Set plot titles and labels
    a.set_title(title, fontsize=20)
    a.set_xlabel(xlabel, fontsize=18)
    a.set_ylabel(ylabel, fontsize=18)
    if y_col == 'sample_depth':  # Invert y-axis for depth
        a.invert_yaxis()

plt.tight_layout()  # Adjust layout to not overlap text
plt.show()




data_new = pd.DataFrame()
data_new['sample_ID'] = ['1_1', '1_2', '2_1', '2_2', '2_3', '2_4', '2_5', '2_6', '2_7', '2_8', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6', '5_1', '5_2', '5_3', '5_4', '6_1', '6_2', '6_3', '6_4', '6_5', '6_6', '6_7', '6_8', '6_9', '6_10', '6_11']
data_new['sample_depth'] = [35, 80, 32, 50, 63, 82, 91, 130, 142, 163, 33, 53, 62, 77, 36, 52, 67, 85, 152, 180, 36, 53, 76, 92, 42, 68, 84, 145, 150, 158, 178, 184, 200, 245, 277]
data_new['Kappa_at_0.46kHz_LF_(vol.)'] = [768, 67, 892, 1041, 854, 938, 1095, 707, 805, 9, 957, 164, 95, 3, 777, 582, 23, 12, 10, 22, 968, 782, 512, 131, 745, 484, 448, 387, 448, 274, 510, 630, 58, 17, 16]



# Function to plot specific drilling profiles on separate graphs within the same figure
def plot_profiles_separate(data_frame, profiles, title):
    fig, ax = plt.subplots(nrows=1, ncols=len(profiles), figsize=(15, 5))
    y_range = (0, 3)  # Replace with your desired X axis range
    x_range = (0, 15)
    for i, profile in enumerate(profiles):
        profile_data = data_frame[data_frame['sample_ID'].str.startswith(str(profile) + '_')]
        ax[i].plot(profile_data['Kappa_at_0.46kHz_LF_(vol.)']/100, profile_data['sample_depth']/100, color='#000000')
        ax[i].scatter(profile_data['Kappa_at_0.46kHz_LF_(vol.)']/100, profile_data['sample_depth']/100, color='#000000')
        ax[i].set_xlim(x_range)
        ax[i].set_ylim(y_range)
        ax[i].invert_yaxis()
        ax[i].tick_params(axis='both', which='major', labelsize=20)  # Set the fontsize here
        ax[i].set_ylabel('Sample Depth [m]', fontsize=22)
        ax[i].set_xlabel('$\kappa_{lf}$ x $10^{-4}$ [SI units]', fontsize=22)
        ax[i].set_title(f'Profile {profile}', fontsize=28)
        ax[i].legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# First set of profiles: 4, 2, and 5
plot_profiles_separate(data_new, [4, 2, 5, 1], '')

# Second set of profiles: 3, 6, 2, and 1
plot_profiles_separate(data_new, [3, 6, 2, 1], '')


