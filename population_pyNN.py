"""


"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bluepyopt as bpop
import bluepyopt.ephys as ephys
import bluepyopt.ephys_pyNN as epyn

import logging
logging.basicConfig(filename='population_pyNN.log', filemode='w', level=logging.DEBUG)

PLOT_FIGURES = True
POP_SIZE = 10
N_GEN = 10

# patch the bpop.optimisations module

def evaluate_fitnesses(toolbox, individuals):
    # individuals is a list of lists, each list containing the parameters for one individual
    return toolbox.evaluate(individuals)

bpop.optimisations.evaluate_fitnesses = evaluate_fitnesses



parameters = {'tau_refrac': [0.1, 10.0], 'a': [0.0, 50.0], 'tau_m': [1.0, 50.0],
              'cm': [0.1, 10.0], 'delta_T': [0.1, 10.0], 'v_thresh': [-70.0, -41.0],
              'b': [0.0, 1.0], 'v_reset': [-80.0, -50.0], 'tau_w': [10.0, 1000.0],
              'v_rest': [-80.0, -50.0]}

parameter_objects = [ephys.parameters.ArrayParameter(name=name, bounds=bounds, frozen=False)
                     for name, bounds in parameters.items()]


simple_population = epyn.models.PyNNPopulationModel(name='simple_population',
                                                    size=POP_SIZE,
                                                    celltype='EIF_cond_exp_isfa_ista',
                                                    params=parameter_objects,
                                                    initial_values={'v': -70.0, 'w': 0.0})

sweep_protocols = []
for protocol_name, amplitude in [('step1', 0.4), ('step2', 2.0)]:
    stim = epyn.stimuli.PyNNSquarePulse(
                step_amplitude=amplitude,
                step_delay=100,
                step_duration=50,
                total_duration=200)
    rec = epyn.recordings.PyNNRecording(
            name='%s.v' % protocol_name,
            variable='v')
    protocol = ephys.protocols.SweepProtocol(protocol_name, [stim], [rec])
    sweep_protocols.append(protocol)
twostep_protocol = ephys.protocols.SequenceProtocol('twostep', protocols=sweep_protocols)


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

responses = twostep_protocol.run(cell_model=simple_population,
                                 param_values=default_params,
                                 sim=simulator)


def plot_responses(responses, channels):
    plt.subplot(2,1,1)
    for channel in channels:
        plt.plot(responses['step1.v'][channel]['time'], responses['step1.v'][channel]['voltage'], label='step1')
    plt.legend()
    plt.subplot(2,1,2)
    for channel in channels:
        plt.plot(responses['step2.v'][channel]['time'], responses['step2.v'][channel]['voltage'], label='step2')
    plt.legend()
    plt.tight_layout()

if PLOT_FIGURES:
    plot_responses(responses, [0, 3])
    plt.savefig("population_pyNN.png")


###
import efel

def features(time, voltage, stim_limits, feature_names):
    start, end = stim_limits
    traces = [{'T': time, 'V': voltage, 'stim_start': [start], 'stim_end': [end]}]
    return efel.getFeatureValues(traces, feature_names)

print(features(responses['step1.v'][0]['time'],
               responses['step1.v'][0]['voltage'],
               (100.0, 150.0),
               ['Spikecount', 'voltage_base']))
print(features(responses['step2.v'][0]['time'],
               responses['step2.v'][0]['voltage'],
               (100.0, 150.0),
               ['Spikecount', 'voltage_base']))
###



efel_feature_means = {'step1': {'Spikecount': 1}, 'step2': {'Spikecount': 5}}

objectives = []

for protocol in sweep_protocols:
    stim_start = protocol.stimuli[0].step_delay
    stim_end = stim_start + protocol.stimuli[0].step_duration
    for efel_feature_name, mean in efel_feature_means[protocol.name].iteritems():
        feature_name = '%s.%s' % (protocol.name, efel_feature_name)
        feature = ephys.efeatures.eFELFeature(
                    feature_name,
                    efel_feature_name=efel_feature_name,
                    recording_names={'': '%s.v' % protocol.name},
                    stim_start=stim_start,
                    stim_end=stim_end,
                    exp_mean=mean,
                    exp_std=0.05 * mean)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            feature)
        objectives.append(objective)

score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)


pop_evaluator = epyn.evaluators.PopulationEvaluator(
        cell_model=simple_population,
        param_names=[p.name for p in parameter_objects],
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc,
        isolate_protocols=False,
        sim=simulator)

print(pop_evaluator.evaluate_with_dicts(default_params))


# ==========

optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=pop_evaluator,
        offspring_size=POP_SIZE)

final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=N_GEN)

#print('Final population: ', final_pop)

best_ind = hall_of_fame[0]
print('Best individual: ', best_ind)
print('Fitness values: ', best_ind.fitness.values)

best_ind_dict = pop_evaluator.param_dict(best_ind)
print(pop_evaluator.evaluate_with_dicts(best_ind_dict))

if PLOT_FIGURES:
    plt.clf()
    plot_responses(twostep_protocol.run(cell_model=simple_population, param_values=best_ind_dict, sim=simulator),
                   channels=[0])
    plt.savefig("population_pyNN_best.png")


gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
for gen, fitness in zip(gen_numbers, min_fitness):
    print(gen, fitness)
if PLOT_FIGURES:
    plt.clf()
    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.xlabel('generation #')
    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    plt.ylim(0.9*min(min_fitness), 1.1 * max(min_fitness))
    plt.savefig("population_pyNN_fitness.png")

