import matplotlib
import orbipy as op
import numpy as np
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt

"""
There are two "data structures" used here:
- contour_data => a list [ [[x,y,z], [x,y,z], [x,y,z], ...], [[x,y,z], [x,y,z], [x,y,z], ...], ... ],
        an array of contour lines, consisting of points.
- points_data => a list [ [x,y,z], [x,y,z], [x,y,z], ... ], array of all points.

Former is easier to work with and is the output of orbits calculation methods,
    the latter is easier to visualize (and is a numpy array)
"""

def icm_contour_visualizer(data, figsize_x=9, figsize_y=7, points_size=3, l1=True, moon=True, convert_to_isu=False):
    """
    Visualizes x-z Initial Conditions Map data, of type "contour_data" .
    :param data: list of type "contour_data"
    :param l1: boolean, True - l1, False - l2
    :param moon: boolean, True - Moon, False - Earth
    :param convert_to_isu: boolean, convert graph's units of measurements to isu (kms, km/s)
    :return: matplotlib.pyplot.figure
    """
    # Data parsing

    # converts to points_data
    data_points = []
    for line in data[:]: # here you can specify what contour lines to draw, for example "[180:230]"
        for point in line:
            data_points.append(point)

    data_points = np.array(data_points)

    # Plot configuring
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # plotting of data
    scatter = ax.scatter(data_points[:,0], data_points[:,1], c=data_points[:,2], cmap='magma', s=points_size)
    fig.gca().set_aspect('equal', adjustable='box') # ensures equal scales of axis

    cbar = fig.colorbar(scatter, ax=ax)

    if moon:
        model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    else:
        model = op.crtbp3_model('Sun-Earth (default)', integrator=op.dopri5_integrator(), stm=True)

    if convert_to_isu:
        if moon:
            scale_d = 384.4  # thousands kms, distance from Earth to Moon
            scale_v = 1.0251
            ax.set_xlabel(r"x, $10^3$ км")
            ax.set_ylabel(r"y, $10^3$ км")
        else:
            scale_d = 151.4  # millions kms, distance from Earth to Sun
            scale_v = 1.0251
            ax.set_xlabel(r"x, $10^6$ км")
            ax.set_ylabel(r"y, $10^6$ км")

        km_formatter = FuncFormatter(lambda value, pos: f"{value * scale_d:.2f}")
        v_formatter = FuncFormatter(lambda value, v: f"{value * scale_v:.2f}")

        cbar.set_label("км/с")

        # Apply formatter to the colorbar ticks
        cbar.ax.yaxis.set_major_formatter(v_formatter)
        # Apply to both axes
        ax.xaxis.set_major_formatter(km_formatter)
        ax.yaxis.set_major_formatter(km_formatter)

    else:
        cbar.set_label(r"$v_y$")
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

    ax.scatter(1, 0) # Massive object №2
    if moon:
        ax.annotate("Луна", (0.99, 0.01))
    else:
        ax.annotate("Земля", (0.99, 0.01))

    if l1:
        ax.scatter(model.L1, 0)
        ax.annotate("L1", (model.L1, -0.0115))
    else:
        ax.scatter(model.L2, 0)
        ax.annotate("L2", (model.L2, -0.0115))

    return fig


def icm_points_visualizer(data, figsize_x=9, figsize_y=7, points_size=3, l1=True, moon=True, convert_to_isu=False):
    """
    Visualizes x-z Initial Conditions Map data, of type "points_data" .
    :param data: list of type "points_data"
    :param l1: boolean, True - l1, False - l2
    :param moon: boolean, True - Moon, False - Earth
    :param convert_to_isu: boolean, convert graph's units of measurements to isu (kms, km/s)
    :return: matplotlib.pyplot.figure
    """

    data_points = np.array(data)

    # Plot configuring
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    # plotting of data
    scatter = ax.scatter(data_points[:,0], data_points[:,1], c=data_points[:,2], cmap='magma', s=points_size)
    fig.gca().set_aspect('equal', adjustable='box') # ensures equal scaling of axis

    cbar = fig.colorbar(scatter, ax=ax)

    if moon:
        model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    else:
        model = op.crtbp3_model('Sun-Earth (default)', integrator=op.dopri5_integrator(), stm=True)

    if convert_to_isu:
        if moon:
            scale_d = 384.4  # thousands kms, distance from Earth to Moon
            scale_v = 1.0251
            ax.set_xlabel(r"x, $10^3$ км")
            ax.set_ylabel(r"y, $10^3$ км")
        else:
            scale_d = 151.4  # millions kms, distance from Earth to Sun
            scale_v = 1.0251
            ax.set_xlabel(r"x, $10^6$ км")
            ax.set_ylabel(r"y, $10^6$ км")

        km_formatter = FuncFormatter(lambda value, pos: f"{value * scale_d:.2f}")
        v_formatter = FuncFormatter(lambda value, v: f"{value * scale_v:.2f}")

        cbar.set_label("км/с")

        # Apply formatter to the colorbar ticks
        cbar.ax.yaxis.set_major_formatter(v_formatter)
        # Apply to both axes
        ax.xaxis.set_major_formatter(km_formatter)
        ax.yaxis.set_major_formatter(km_formatter)

    else:
        cbar.set_label(r"$v_y$")
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

    ax.scatter(1, 0)
    if moon:
        ax.annotate("Луна", (0.99, 0.01))
    else:
        ax.annotate("Земля", (0.99, 0.01))

    if l1:
        ax.scatter(model.L1, 0)
        ax.annotate("L1", (model.L1, -0.0115))
    else:
        ax.scatter(model.L2, 0)
        ax.annotate("L2", (model.L2, -0.0115))

    return fig


def icm_contour_comparison(data_1, data_2, label_1="data_1", label_2="data_2", figsize_x=9, figsize_y=7, points_size=3, l1=True, moon=True, convert_to_isu=False):
    """
    Overlays two Initial Condition Maps (of datatype "contour_data") on a single matplotlib graph, for easier comparison
    :param data_1: first icm map, of type "contour_data"
    :param data_2: second icm map, of type "contour_data"
    :param label_1: label of the first map
    :param label_2: label of the second map
    :return: matplotlib.pyplot.figure
    """
    # Data parsing

    data_points_1 = []
    data_points_2 = []
    for line in data_1[:]: # here you can specify what contour lines to draw, for example "[180:230]"
        for point in line:
            data_points_1.append(point)

    for line in data_2[:]:  # here you can specify what contour lines to draw, for example "[180:230]"
        for point in line:
            data_points_2.append(point)

    data_points_1 = np.array(data_points_1)
    data_points_2 = np.array(data_points_2)

    # Plot configuring

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    ax.scatter(data_points_1[:,0], data_points_1[:,1], s=points_size, label=label_1)
    ax.scatter(data_points_2[:,0], data_points_2[:,1], s=points_size, label=label_2)
    fig.gca().set_aspect('equal', adjustable='box')

    if moon:
        model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    else:
        model = op.crtbp3_model('Sun-Earth (default)', integrator=op.dopri5_integrator(), stm=True)

    if convert_to_isu:
        if moon:
            scale_d = 384.4  # thousands kms, distance from Earth to Moon
            ax.set_xlabel(r"x, $10^3$ км")
            ax.set_ylabel(r"y, $10^3$ км")
        else:
            scale_d = 151.4  # millions kms, distance from Earth to Sun
            ax.set_xlabel(r"x, $10^6$ км")
            ax.set_ylabel(r"y, $10^6$ км")

        km_formatter = FuncFormatter(lambda value, pos: f"{value * scale_d:.2f}")

        # Apply to both axes
        ax.xaxis.set_major_formatter(km_formatter)
        ax.yaxis.set_major_formatter(km_formatter)

    else:
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

    ax.scatter(1, 0)
    if moon:
        ax.annotate("Луна", (0.99, 0.01))
    else:
        ax.annotate("Земля", (0.99, 0.01))

    if l1:
        ax.scatter(model.L1, 0)
        ax.annotate("L1", (model.L1, -0.0115))
    else:
        ax.scatter(model.L2, 0)
        ax.annotate("L2", (model.L2, -0.0115))

    ax.legend()

    return fig


def icm_points_comparison(data_1, data_2, label_1="data_1", label_2="data_2", figsize_x=9, figsize_y=7, points_size=3, l1=True, moon=True, convert_to_isu=False):
    """
    Overlays two Initial Conditions Maps (of datatype "points_data") on a single matplotlib graph, for easier comparison
    :param data_1: first icm map, of type "points_data"
    :param data_2: second icm map, of type "points_data"
    :param label_1: label of the first map
    :param label_2: label of the second map
    :return: matplotlib.pyplot.figure
    """
    # Data parsing

    data_points_1 = np.array(data_1)
    data_points_2 = np.array(data_2)

    # Plot configuring

    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    ax.scatter(data_points_1[:,0], data_points_1[:,1], s=points_size, label=label_1)
    ax.scatter(data_points_2[:,0], data_points_2[:,1], s=points_size, label=label_2)
    fig.gca().set_aspect('equal', adjustable='box')

    if moon:
        model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    else:
        model = op.crtbp3_model('Sun-Earth (default)', integrator=op.dopri5_integrator(), stm=True)

    if convert_to_isu:
        if moon:
            scale_d = 384.4  # thousands kms, distance from Earth to Moon
            ax.set_xlabel(r"x, $10^3$ км")
            ax.set_ylabel(r"y, $10^3$ км")
        else:
            scale_d = 151.4  # millions kms, distance from Earth to Sun
            ax.set_xlabel(r"x, $10^6$ км")
            ax.set_ylabel(r"y, $10^6$ км")

        km_formatter = FuncFormatter(lambda value, pos: f"{value * scale_d:.2f}")


        # Apply to both axes
        ax.xaxis.set_major_formatter(km_formatter)
        ax.yaxis.set_major_formatter(km_formatter)

    else:
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

    ax.scatter(1, 0)
    if moon:
        ax.annotate("Луна", (0.99, 0.01))
    else:
        ax.annotate("Земля", (0.99, 0.01))

    if l1:
        ax.scatter(model.L1, 0)
        ax.annotate("L1", (model.L1, -0.0115))
    else:
        ax.scatter(model.L2, 0)
        ax.annotate("L2", (model.L2, -0.0115))

    ax.legend()

    return fig


def vcsm_contour_visuzalizer(data, show_fails=True, filter=False, figsize_x=9, figsize_y=7, points_size=3, l1=True, moon=True, convert_to_isu=False):
    """
    Visualizes Velocity Corrections Sum Map data for an icm map. typeof(data) is "contour_data".
    :param data: dv sum data, type="contour_data", [ [ [[x,y,z], dv_sum], [[x,y,z], dv_sum], ... ], ... ]
    :param show_fails: boolean, show points that were failed to calculate (returned 10 as dv_sum)
    :param filter: boolean, filter or not the dv_sum data (<= 10^-6)
    :return: matplotlib.pyplot.figure
    """
    data_points = []

    for line in data[:]:
        for point in line:
            tmp = [*point[0], point[1]] # flattens the dv_sum point, from [[x,y,z],dv_sum] to [x,y,z,dv_sum]
            if not show_fails: # if flag, do not show failed points
                if tmp[-1] == 10:
                    continue
            data_points.append(tmp)

    data_points = np.array(data_points)
    data_points_surface = data_points
    if filter: # if flag, filter out points with dv_sum > 10^-6
        data_points_surface = data_points[np.where(data_points[:, 3] <= 10**-6)]

    # Plot configuring
    fig, ax = plt.subplots(figsize=(figsize_x, figsize_y))

    scatter = ax.scatter(data_points_surface[:, 0], data_points_surface[:, 1], c=data_points_surface[:, 3], cmap='magma', s=points_size,
                norm=matplotlib.colors.LogNorm())

    fig.gca().set_aspect('equal', adjustable='box')

    fig.colorbar(scatter, ax=ax)

    if moon:
        model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    else:
        model = op.crtbp3_model('Sun-Earth (default)', integrator=op.dopri5_integrator(), stm=True)

    if convert_to_isu:
        if moon:
            scale_d = 384.4  # thousands kms, distance from Earth to Moon
            ax.set_xlabel(r"x, $10^3$ км")
            ax.set_ylabel(r"y, $10^3$ км")
        else:
            scale_d = 151.4  # millions kms, distance from Earth to Sun
            ax.set_xlabel(r"x, $10^6$ км")
            ax.set_ylabel(r"y, $10^6$ км")

        km_formatter = FuncFormatter(lambda value, pos: f"{value * scale_d:.2f}")

        # Apply to both axes
        ax.xaxis.set_major_formatter(km_formatter)
        ax.yaxis.set_major_formatter(km_formatter)

    else:
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")

    ax.scatter(1, 0)
    if moon:
        ax.annotate("Луна", (0.99, 0.01))
    else:
        ax.annotate("Земля", (0.99, 0.01))

    if l1:
        ax.scatter(model.L1, 0)
        ax.annotate("L1", (model.L1, -0.0115))
    else:
        ax.scatter(model.L2, 0)
        ax.annotate("L2", (model.L2, -0.0115))

    return fig

def apply_vcsm_filter_to_icm_data(vcsm, data):
    """
    Applies Velocity Corrections Sum Map data to Initial Conditions Map.
    Only leaves points on the icm map which pass the condition of dv_sum <= 10^-6.
    :param vcsm: vcsm data, format "contour_data"
    :param data: icm data, format "contour_data"
    :return: "contour_data" array.
    """
    filtered_contour_data = []
    index = 0

    # Searches for the index of the vcsm inside data
    # (for example, index=180 would mean that data[180:] contains vcsm data.
    for i, line in enumerate(data):
        if len(line) == 0: #checks for empty contour lines
            continue
        if line[0] == vcsm[0][0][0]:
            index = i
            break

    for line in data[:index]: # Append first index-1 contour lines to the output array. They are not filtered by definition.
        filtered_contour_data.append(np.array(line))

    # filtering
    for line_vcsm, line_data in zip(vcsm, data[index:]):
        line_vcsm_np = np.array([[*point[0], point[1]] for point in line_vcsm]) # flatten and convert to numpy array
        line_data_np = np.array(line_data) # convert to numpy array

        line_filtered = line_vcsm_np[np.where(line_vcsm_np[:, 3] > 10 ** -6)][:, :3] # filter dv_sum > 10^-6, outputing array without dv_sum data
        matches = (line_data_np[:, None, :] == line_filtered[None, :, :]).all(axis=2).any(axis=1) # creating mask array, match if dv_sum > 10^-6
        filtered = line_data_np[~matches] # masking out unwanted data (~masked is "not masked", so it leaves only data with dv_sum <= 10^-6)
        filtered_contour_data.append(filtered)

    return filtered_contour_data
