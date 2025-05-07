import argparse
import sys
import orbipy as op
import numpy as np
from scipy.optimize import bisect
import pickle


def calculate_next_point_with_same_jacoby_constant(correction, previous_state, cj, previous_alpha_degrees, dr, model):
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

    res = bisect(find_alpha, previous_alpha_degrees - 40, previous_alpha_degrees + 40,
        xtol=1e-6, maxiter=100, full_output=True)

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

def calculate_contour_line_of_jakobi_constant(model, initial_state, alpha, dr):
    Cj = model.jacobi(initial_state)

    left = op.eventX(model.L1 - 33 * one_thousand_kms)
    right = op.eventX(1)
    correction = op.border_correction(model, op.y_direction(), [left], [right])

    contour = []

    current_state = initial_state.copy()
    current_alpha = alpha

    steps = 0
    while True:
        # stop if z dips below zero
        if current_state[2] <= 0:
            break
        contour.append(current_state[[0, 2, 4]].tolist())

        try:
            new_state, current_alpha = calculate_next_point_with_same_jacoby_constant(
                correction, current_state, Cj, current_alpha, dr, model
            )
        except (RuntimeError, ValueError) as e:
            print(f"Step {steps} failed: {e}")
            break

        steps += 1
        current_state = new_state

    return contour


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute contour lines of constant Jacobi around L1 in the Earth-Moon CR3BP."
    )
    parser.add_argument(
        '--offset', '-o',
        type=int,
        required=True,
        help="Index of starting point in the ZVL data array."
    )
    parser.add_argument(
        '--size', '-s',
        type=int,
        required=True,
        help="Number of contours to compute starting from offset."
    )
    parser.add_argument(
        '--output', '-f',
        type=str,
        default=None,
        help = "Optional output filename. By default uses offset and size."
    )
    args = parser.parse_args()

    # load zero-velocity line (ZVL) points
    data = np.load('zvl_3.npy')

    # set up CRTBP model and length scale
    model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    one_thousand_kms = (1 - model.L1) / 61.350
    dr = 1 * one_thousand_kms

    # determine the slice of ZVL points
    start = args.offset
    end = start + args.size
    if start < 0 or end > len(data):
        end = len(data)
        parser.error(f"Offset and size out of range: must satisfy 0 <= offset < len(data) and offset+size <= {len(data)}.")

    # prepare indices and alphas
    indices = range(start, end)
    alphas = [0, 180]

    results = []
    filename = f"contour_offset{start}_size{args.size}.pickle"

    for idx in indices:
        for alpha in alphas:
            print(f"Computing contour for ZVL index={idx}, alpha={alpha}...")
            # build initial state from ZVL data
            init = model.get_zero_state()
            init[[0, 2]] = data[idx][0], data[idx][1]
            # vy inherited from zero state

            contour = calculate_contour_line_of_jakobi_constant(model, init, alpha, dr)
            results.append(contour)

            with open(filename, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done. Results written to {filename}")
