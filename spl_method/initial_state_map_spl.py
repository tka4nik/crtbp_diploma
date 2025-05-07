import orbipy as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import pickle


def calculate_next_point_with_same_jacoby_constant(correction, previous_state, cj, previous_alpha_degrees):
    def find_alpha(alpha_degrees):
        state = model.get_zero_state().copy()
        x = previous_state[0] + np.cos(np.radians(alpha_degrees))*dr
        z = previous_state[2] + np.sin(np.radians(alpha_degrees))*dr
        state[[0,2]] = x, z
        state[4] = previous_state[4]
        np.set_printoptions(precision=10)
        # print(f"    {state[[0, 2, 4]]}")
        velocity = correction.calc_dv(0, state)
        state += velocity
        cj_new = model.jacobi(state)
        return cj - cj_new

    res = bisect(find_alpha, previous_alpha_degrees - 30, previous_alpha_degrees + 30,
        xtol=1e-6, maxiter=100,
        full_output=True)

    target_alpha_degrees = res[0]
    np.set_printoptions(precision=10)
    print(f"    cj = {cj}; alpha = {target_alpha_degrees}")
    r = model.get_zero_state().copy()
    x = previous_state[0] + np.cos(np.radians(target_alpha_degrees))*dr
    z = previous_state[2] + np.sin(np.radians(target_alpha_degrees))*dr
    vy = previous_state[4]
    r[[0, 2, 4]] = x, z, vy
    print(f"   resulted point: {r[:3]}")
    r += correction.calc_dv(0, r)
    return r, target_alpha_degrees

def calculate_contour_line_of_jakobi_constant(model, initial_state, alpha, points):
    Cj = model.jacobi(initial_state)

    points.append(initial_state[[0, 2, 4]].tolist())

    while points[-1][1] > 0:
        left = op.eventSPL(model, Cj, accurate=False)
        right = op.eventSPL(model, Cj, accurate=False, left=False)

        correction = op.border_correction(model, op.y_direction(), left, right, dv0=0.03, maxt=1000.)

        try:
            next_point_with_same_jakoby_constant, alpha = calculate_next_point_with_same_jacoby_constant(correction, initial_state, Cj, alpha)
        except (RuntimeError, ValueError) as e:
            print(f"next point from {points[-1]} failed! {e}")
            break
        initial_state = next_point_with_same_jakoby_constant
        points.append(initial_state[[0, 2, 4]].tolist())
        print(points[-1])
    print("==== Finished ====")


if __name__ == '__main__':
    data = np.load('../data/contour_points/zvl/zvl_4_sparse.npy')
    model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    one_thousand_kms = (1-model.L1) / 61.350

    dr = 1*one_thousand_kms
    points = []

    initial_state_points = np.array(np.meshgrid(np.arange(5, 134), [0, 180])).T.reshape(-1, 2).tolist()
    print(initial_state_points)

    for point in initial_state_points:
        alpha = point[1]
        initial_state = model.get_zero_state()
        initial_state[[0,2]] = data[point[0]][0], data[point[0]][1]

        new_points = []
        print(f"ZVL: {point}")
        calculate_contour_line_of_jakobi_constant(model, initial_state, alpha, new_points)
        print(new_points)
        points.append(new_points)
        with open("../data/contour_points/contour_points_data_spl_4.pickle", 'wb') as handle:
            pickle.dump(points, handle, protocol=pickle.HIGHEST_PROTOCOL)
