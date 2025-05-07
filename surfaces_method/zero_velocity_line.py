import orbipy as op
import numpy as np
from scipy.optimize import bisect


def zero_velocity_correction(correction, initial_state, previous_alpha_degrees, dr):
    def find_alpha(alpha_degrees):
        state = model.get_zero_state().copy()
        x = initial_state[0] + np.cos(np.radians(alpha_degrees)) * dr
        z = initial_state[2] + np.sin(np.radians(alpha_degrees)) * dr
        state[[0, 2]] = x, z
        v = correction.calc_dv(0, state)
        return v[4]

    res = bisect(find_alpha, previous_alpha_degrees - 30, previous_alpha_degrees + 30,
                 xtol=1e-6, maxiter=100,
                 full_output=True)

    target_alpha_degrees = res[0]
    print(f"    alpha {target_alpha_degrees}")
    r = np.zeros_like(initial_state)
    x = np.cos(np.radians(target_alpha_degrees)) * dr
    z = np.sin(np.radians(target_alpha_degrees)) * dr
    r[[0, 2]] = x, z
    return r, target_alpha_degrees


def zero_velocity_line_calculation(model, output):
    alpha = 90
    left = op.eventX(model.L1 - 33 * one_thousand_kms)
    right = op.eventX(1)
    correction = op.border_correction(model, op.y_direction(), left, right)

    dr = 0.3 * one_thousand_kms
    initial_state = model.get_zero_state()
    initial_state[[0, 2]] = [model.L1, 0]
    output.append(initial_state[[0, 2]])

    for i in range(500):
        zero_velocity_point_correction, alpha = zero_velocity_correction(correction, initial_state, alpha, dr)
        initial_state += zero_velocity_point_correction
        output.append(initial_state[[0, 2]])
        print(f'{i}: {output[-1]}')
    print("==== Finished ====")

if __name__ == '__main__':
    model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    one_thousand_kms = (1 - model.L1) / 61.350
    output = []
    zero_velocity_line_calculation(model, output)
    zero_velocity_line = np.array(output)
    np.save('../data/contour_points/zvl/zvl_4.npy', zero_velocity_line)





