"""
Example of fitting the AdEx model to a recording from a NEURON model
of a L5 pyramidal neuron using BluePyOpt.

Andrew Davison, CNRS, April 2016

"""

from __future__ import print_function
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bluepyopt as bpop
import bluepyopt.ephys as ephys
import bluepyopt.ephys_pyNN as epyn
import efel

logging.basicConfig(filename='l5pyr_adex_example.log', filemode='w', level=logging.DEBUG)

PLOT_FIGURES = True
POP_SIZE = 100
N_GEN = 20
FEATURE_NAMES = ['voltage_base', 'inv_time_to_first_spike', 'inv_first_ISI',
                 'Spikecount', 'AHP_depth_abs', 'AP_amplitude']


# We need to patch the bpop.optimisations module

def evaluate_fitnesses(toolbox, individuals):
    # individuals is a list of lists, each list containing the parameters for one individual
    return toolbox.evaluate(individuals)

bpop.optimisations.evaluate_fitnesses = evaluate_fitnesses


# Extract the target features from recordings from a previous simulation
# of a detailed pyramidal neuron model by the Blue Brain Project.

original_datafiles = {
    "step1": "L5_TTPC1_cADpyr232_1/python_recordings/soma_voltage_step1.dat",
    "step2": "L5_TTPC1_cADpyr232_1/python_recordings/soma_voltage_step2.dat",
    "step3": "L5_TTPC1_cADpyr232_1/python_recordings/soma_voltage_step3.dat",
}
original_data = dict((label, np.loadtxt(filename).T)
                     for label, filename in original_datafiles.items())
sweep_timepoints = (0, 700, 2700, 3000)
sweep_amplitudes = {
    'step1': 0.55425,
    'step2': 0.6004375,
    'step3': 0.646625
}


def features(data, stim_limits, feature_names):
    time, voltage = data
    start, end = stim_limits
    traces = [{'T': time, 'V': voltage, 'stim_start': [start], 'stim_end': [end]}]
    return efel.getFeatureValues(traces, feature_names)


efel_feature_means = {}
for protocol_name, data in original_data.items():
    features_original_model = features(data, sweep_timepoints[1:-1], FEATURE_NAMES)
    efel_feature_means[protocol_name] = {}
    for trace_results in features_original_model:
        # trace_result is a dictionary, with as keys the requested eFeatures
        for feature_name, feature_values in trace_results.items():
            efel_feature_means[protocol_name][feature_name] = np.mean(feature_values)

print("Features of original data:")
print(efel_feature_means)


# Define the AdEx model population

parameters = {'tau_refrac': [0.1, 10.0], 'a': [0.0, 50.0], 'tau_m': [1.0, 50.0],
              'cm': [0.1, 10.0], 'delta_T': [0.1, 10.0], 'v_thresh': [-70.0, -41.0],
              'b': [0.0, 1.0], 'v_reset': [-80.0, -50.0], 'tau_w': [10.0, 1000.0],
              'v_rest': [-80.0, -50.0]}

parameter_objects = [ephys.parameters.ArrayParameter(name=name, bounds=bounds, frozen=False)
                     for name, bounds in parameters.items()]

adex_population = epyn.models.PyNNPopulationModel(name='adex_population',
                                                  size=POP_SIZE,
                                                  celltype='EIF_cond_exp_isfa_ista',
                                                  params=parameter_objects,
                                                  initial_values={'v': -70.0, 'w': 0.0})

# Define the stimulation protocol

sweep_protocols = []
for protocol_name, amplitude in sweep_amplitudes.items():
    stim = epyn.stimuli.PyNNCurrentPlayStimulus(
                time_points=sweep_timepoints[:-1],
                current_points=(-0.247559, amplitude, -0.247559),
                total_duration=sweep_timepoints[-1])
    rec = epyn.recordings.PyNNRecording(
            name='%s.v' % protocol_name,
            variable='v',
            artificial_ap=30.0)
    protocol = ephys.protocols.SweepProtocol(protocol_name, [stim], [rec])
    sweep_protocols.append(protocol)
threestep_protocol = ephys.protocols.SequenceProtocol('threestep', protocols=sweep_protocols)

simulator = epyn.simulators.PyNNSimulator('nest')

default_params = {'tau_refrac': np.random.normal(2.0, 0.4, size=POP_SIZE),
                  'a': np.random.normal(4.0, 1.0, size=POP_SIZE),
                  'tau_m': np.random.normal(10.0, 1.0, size=POP_SIZE),
                  'cm': np.random.normal(0.5, 0.1, size=POP_SIZE),
                  'delta_T': np.random.normal(2.0, 0.4, size=POP_SIZE),
                  'v_thresh': -np.random.normal(50.0, 1.0, size=POP_SIZE),
                  'b': np.random.normal(0.1, 0.01, size=POP_SIZE),
                  'v_reset': np.random.normal(-70.0, 1.0, size=POP_SIZE),
                  'tau_w': np.random.normal(100.0, 20.0, size=POP_SIZE),
                  'v_rest': np.random.normal(-70.0, 2.0, size=POP_SIZE)}

# Test run

responses = threestep_protocol.run(cell_model=adex_population,
                                   param_values=default_params,
                                   sim=simulator)


def plot_responses(responses, channels):
    plt.subplot(3, 1, 1)
    plt.plot(original_data['step1'][0], original_data['step1'][1], label='original')
    for channel in channels:
        plt.plot(responses['step1.v'][channel]['time'], responses['step1.v'][channel]['voltage'], label='channel {0}'.format(channel))
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(original_data['step2'][0], original_data['step2'][1], label='original')
    for channel in channels:
        plt.plot(responses['step2.v'][channel]['time'], responses['step2.v'][channel]['voltage'], label='channel {0}'.format(channel))
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(original_data['step3'][0], original_data['step3'][1], label='original')
    for channel in channels:
        plt.plot(responses['step3.v'][channel]['time'], responses['step3.v'][channel]['voltage'], label='channel {0}'.format(channel))
    plt.legend()
    plt.tight_layout()

if PLOT_FIGURES:
    plot_responses(responses, [0])
    plt.savefig("l5pyr_adex_example.png")


# Define fitness measure

objectives = []
for protocol in sweep_protocols:
    for efel_feature_name, mean in efel_feature_means[protocol.name].items():
        feature_name = '%s.%s' % (protocol.name, efel_feature_name)
        feature = ephys.efeatures.eFELFeature(
                    feature_name,
                    efel_feature_name=efel_feature_name,
                    recording_names={'': '%s.v' % protocol.name},
                    stim_start=sweep_timepoints[1],
                    stim_end=sweep_timepoints[2],
                    exp_mean=mean,
                    exp_std=0.05 * mean)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            feature)
        objectives.append(objective)

score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)

# Configure and run optimisation

pop_evaluator = epyn.evaluators.PopulationEvaluator(
        cell_model=adex_population,
        param_names=[p.name for p in parameter_objects],
        fitness_protocols={threestep_protocol.name: threestep_protocol},
        fitness_calculator=score_calc,
        sim=simulator)

optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=pop_evaluator,
        offspring_size=POP_SIZE)

final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=N_GEN)

best_ind = hall_of_fame[0]
print('\nBest individual: ')
for p, val in zip(parameter_objects, best_ind):
    print("  {} = {}".format(p.name, val))

print('\nFitness values: ')
best_ind_dict = pop_evaluator.param_dict(best_ind)
responses_best = threestep_protocol.run(cell_model=adex_population,
                                        param_values=best_ind_dict,
                                        sim=simulator)
# todo: calculate features from responses_best
features_best = {}
for label, traces in responses_best.items():
    feat = features((traces[0]['time'], traces[0]['voltage']), sweep_timepoints[1:-1], FEATURE_NAMES)
    for feature_name, value in feat[0].items():
        if value is not None:
            features_best[label.replace('v', feature_name)] = value.mean()
        else:
            features_best[label.replace('v', feature_name)] = value
features_data = {}
for label, feat in efel_feature_means.items():
    for feature_name, value in feat.items():
        features_data["{}.{}".format(label, feature_name)] = value
print("Feature                        Fitness         Original data   Best fit")
print("-"*30 + " " + "-"*15 + " " + "-"*15 + " " + "-"*15)
for feature_name, value in pop_evaluator.evaluate_with_dicts(best_ind_dict)[0].items():
    print("{:<30} {:<15} {:<15} {:<15}".format(feature_name, value, features_data[feature_name], features_best[feature_name]))

if PLOT_FIGURES:
    plt.clf()
    plot_responses(responses_best, channels=[0])
    plt.savefig("l5pyr_adex_example_best.png")

gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
print("\nGeneration   Fitness")
for gen, fitness in zip(gen_numbers, min_fitness):
    print("{:>6}       {}".format(gen, fitness))
if PLOT_FIGURES:
    plt.clf()
    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.xlabel('generation #')
    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    plt.ylim(0.9*min(min_fitness), 1.1 * max(min_fitness))
    plt.savefig("l5pyr_adex_example_fitness.png")

