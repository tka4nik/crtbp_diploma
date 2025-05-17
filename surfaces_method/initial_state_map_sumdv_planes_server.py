import orbipy as op
import numpy as np
import argparse
import pickle

model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)

class zero_correction(op.corrections.base_correction):
    def __init__(self, model, direction):
        super().__init__(model, direction)

    def calc_dv(self, t, s):
        return model.get_zero_state()


def generate_initial_state_map_with_sum_dv(model, contour_line_data, T):
    def find_approximate_half_period(model, state, first_correction, correction):
        event = op.eventY(count=3, terminal=True)
        impulse_correction = op.simple_station_keeping(model, first_correction, correction, rev=np.pi/4,
                                                       events=[event], verbose=False)

        try:
            _ = impulse_correction.prop(0, state, 20)
        except (RuntimeError, ValueError) as e:
            print(f"    Failed to compute half period: {e}; setting default value of np.pi/2")
            return np.pi / 2

        time = impulse_correction.evout[:3, 3]
        half_period = time[-1] - time[-2]
        return half_period

    def impulse_correction_sum_dv(model, state, T):
        print(f"    Calculating dv sum for state[{state}]:")
        initial_state = model.get_zero_state()

        initial_state[0] = state[0]
        initial_state[2] = state[1]
        initial_state[4] = state[2]

        left = op.eventX(model.L1 - 33 * one_thousand_kms)
        right = op.eventX(model.L1 + 55 * one_thousand_kms)
        correction = op.border_correction(model, op.y_direction(), [left], [right])

        first_correction = zero_correction(model, op.unstable_direction(model))

        half_period = find_approximate_half_period(model, initial_state, first_correction, correction)
        n = int(2 * (T // half_period))
        print(f'        half_period: {half_period}, number of corrections: {n}')

        impulse_correction = op.simple_station_keeping(model, first_correction, correction, rev=half_period / 2)
        # quarter period is the most optimal rev time

        try:
            _ = impulse_correction.prop(0, initial_state, n)
        except (RuntimeError, ValueError) as e:
            print(f"    Failed to compute dv sum for state[{state}].")
            return -1

        dv_norms = np.linalg.norm(impulse_correction.dvout[:, 3:6], axis=1)
        return np.sum(dv_norms)

    line_output = []

    zvl_state = model.get_zero_state().copy()
    zvl_state[[0, 2]] = contour_line_data[0][0], contour_line_data[0][1]
    print(f"    Jacobi constant of the contour line: {model.jacobi(zvl_state)}")

    for state in contour_line_data:
        sum_dv = impulse_correction_sum_dv(model, state, T)
        print(f'    dv sum: {sum_dv}')
        line_output.append([state, sum_dv])

    return line_output


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

    args = parser.parse_args()

    # load zero-velocity line (ZVL) points
    with open("contour_points_data.pickle", "rb") as input_file:
        data = pickle.load(input_file)

    # set up CRTBP model and length scale
    model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
    one_thousand_kms = (1 - model.L1) / 61.350
    T = 32

    # determine the slice of ZVL points
    start = args.offset
    end = start + args.size
    if start < 0:
        parser.error(f"Start < 0")

    if end > len(data):
        end = len(data)
        print(f"Offset out of range, end index now equals to {len(data)}.")

    # prepare indices and alphas
    indices = range(start, end)

    results = []
    filename = f"sumdv_offset{start}_size{args.size}.pickle"

    for idx in indices:
        print(f"Computing dv sums for ZVL index={idx}, {"right" if idx % 2 == 0 else "left"} side ...")
        # build initial state from ZVL data
        sumdv_data = generate_initial_state_map_with_sum_dv(model, data[idx], T)
        results.append(sumdv_data)

        with open(filename, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Done. Results written to {filename}")