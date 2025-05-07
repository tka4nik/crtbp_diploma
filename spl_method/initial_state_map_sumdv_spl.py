import orbipy as op
import numpy as np
import pickle

model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)

class zero_correction(op.corrections.base_correction):
    def __init__(self, model, direction):
        super().__init__(model, direction)

    def calc_dv(self, t, s):
        return model.get_zero_state()


def generate_initial_state_map_with_sum_dv(model, data, output_data, T):
    def find_approximate_period(state, first_correction, correction):
        event = op.eventY(count=3, terminal=True)
        impulse_correction = op.simple_station_keeping(model, first_correction, correction, rev=np.pi/4,
                                                       events=[event], verbose=False)

        try:
            _ = impulse_correction.prop(0, state, 20)
        except (RuntimeError, ValueError) as e:
            print(f"!xxxxxx! {state}, Exception: {e}")
            return np.pi / 2

        time = impulse_correction.evout[:3, 3]
        half_period = time[-1] - time[-2]
        return half_period

    def impulse_correction_sum_dv(state, Cj, T):
        initial_state = model.get_zero_state()

        initial_state[0] = state[0]
        initial_state[2] = state[1]
        initial_state[4] = state[2]

        left = op.eventSPL(model, Cj, accurate=False)
        right = op.eventSPL(model, Cj, accurate=False, left=False)

        correction = op.border_correction(model, op.unstable_direction(model), [left], [right], dv0=0.03, maxt=1000.)
        first_correction = zero_correction(model, op.unstable_direction(model))

        half_period = find_approximate_period(initial_state, first_correction, correction)
        print(f'half_period: {half_period}')
        n = int(2 * (T // half_period))

        impulse_correction = op.simple_station_keeping(model, first_correction, correction, rev=half_period / 2)

        try:
            _ = impulse_correction.prop(0, initial_state, n)
        except (RuntimeError, ValueError) as e:
            print(f"!xxxxxx! {state}, Exception: {e}")
            return -1
        dv_norms = np.linalg.norm(impulse_correction.dvout[:, 3:6], axis=1)
        return np.sum(dv_norms)


    for line in data:
        line_output = []
        zvl_state = model.get_zero_state().copy()
        zvl_state[[0, 2]] = line[0][0], line[0][1]
        cj = model.jacobi(zvl_state) # Get jacobi constant from the zero velocity line point

        for state in line:
            print(state)
            sum_dv = impulse_correction_sum_dv(state, cj, T)
            print(f'dv sum: {sum_dv}')
            line_output.append([state, sum_dv])
        output_data.append(line_output)

        with open("../data/contour_points/sumdv_maps/sum_dv_map_spl_51-55.pickle", 'wb') as handle:
            pickle.dump(output_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    one_thousand_kms = (1 - model.L1) / 61.350

    with open(r"../data/contour_points/contour_points_data_spl_3.pickle", "rb") as input_file:
        data = pickle.load(input_file)

    output = []
    generate_initial_state_map_with_sum_dv(model, data[51:], output, 32)