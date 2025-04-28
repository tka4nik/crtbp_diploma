import matplotlib.pyplot as plt
import orbipy as op
import numpy as np
import pandas as pd

model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
halo_orbits = pd.read_csv(f'data/datasets/halo_general_low_l1.csv')

i = 700
zero_state = model.get_zero_state().copy()
zero_state[0] = halo_orbits.iloc[[i]]['x']
zero_state[2] = halo_orbits.iloc[[i]]['z']
zero_state[4] = halo_orbits.iloc[[i]]['v']

one_thousand_kms = (1-model.L1) / 61.350

left = op.eventX(model.L1 - 20 * one_thousand_kms)
right = op.eventX(model.L1 + 60 * one_thousand_kms)
print(left, right)
print(model.jacobi(zero_state))

# event = [op.eventY(count=3)]
# first_correction = op.border_correction(model, op.y_direction(), left, right)
# correction = op.border_correction(model, op.unstable_direction(model), left, right)
# impulse_correction_method = op.simple_station_keeping(model, first_correction, correction, 2.785995635183262, events=event)
#
# df = impulse_correction_method.prop(0.0, zero_state, N=10)
#
# time = impulse_correction_method.evout[:3,3]
# period = (time[-1] - time[-2])*2
# print(period)
#
#
# plotter = op.plotter.from_model(model, length_units='Mm', velocity_units='km/s')
# ax = plotter.plot_proj(df, centers={'x':model.L1})
# plotter.plot_proj(ax=ax, centers={'x':model.L1}, plottables=[plotter.L1], colors='k',ls='',marker='o')
#
# plt.savefig('test.png')



# =====================================
def get_dv_sum_for_dt(model, total_time, dt, initial_state):
    first_correction = op.border_correction(model, op.y_direction(), left, right)
    correction = op.border_correction(model, op.unstable_direction(model), left, right)
    impulse_correction_method = op.strict_station_keeping(model, first_correction, correction, dt, maxdv=1e2)

    df = impulse_correction_method.prop(0.0, initial_state, N=int(total_time // dt))
    dv_norms = np.linalg.norm(impulse_correction_method.dvout[:, 3:6], axis=1)
    return [np.sum(dv_norms), dt], df


model = op.crtbp3_model('Earth-Moon (default)', integrator=op.dopri5_integrator(), stm=True)
halo_orbits = pd.read_csv(f'data/datasets/halo_general_low_l1.csv')

i = 700
period = 2.6357692876211702

zero_state = model.get_zero_state().copy()
zero_state[0] = halo_orbits.iloc[[i]]['x']
zero_state[2] = halo_orbits.iloc[[i]]['z']
zero_state[4] = halo_orbits.iloc[[i]]['v']
one_thousand_kms = (1 - model.L1) / 61.350

left = op.eventX(model.L1 - 10 * one_thousand_kms)
right = op.eventX(model.L1 + 60 * one_thousand_kms)

# total_time = 50*np.pi
total_time = 25 * period

dv_graph_data = []
# dts = [np.pi, np.pi/2, 1.3, np.pi/3, np.pi/4, np.pi/5, np.pi/6, np.pi/7, 0.45, np.pi/8, 0.35, np.pi/9, np.pi/10, np.pi/11]
# dts = np.linspace(np.pi/4, np.pi/8, 15)
# dts = [np.pi/18, np.pi/22,  np.pi/26, np.pi/30]
dts = [period / n for n in range(1, 14)]
print(dts)

for dt in dts:
    try:
        sum, _ = get_dv_sum_for_dt(model, total_time, dt, zero_state.copy())
    except (RuntimeError, ValueError) as e:
        print(e, sep=' ')
        print("++++++ dt = " + str(dt) + " failed!")
        continue
    dv_graph_data.append(sum)
    np.savetxt(f'data/dv_data/dv_graph_data_period_l1-{i}-low-energy.txt', dv_graph_data)
    print(sum)

print(dv_graph_data)
