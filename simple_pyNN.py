"""


"""

try:
    from importlib import import_module
except ImportError:  # Python 2.6
    def import_module(name):
        return __import__(name)
import collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bluepyopt as bpop
import bluepyopt.ephys as ephys


import logging
logging.basicConfig(filename='simple_pyNN.log', filemode='w', level=logging.DEBUG)

PLOT_FIGURES = False


class PyNNModel(object):

    def __init__(self, name, celltype, params=None, initial_values=None):
        self.name = name
        self.celltype = celltype
        self.params = collections.OrderedDict()
        for param in params:
            self.params[param.name] = param
        self.initial_values = initial_values
        self.icell = None
        self.instantiated = False

    def freeze(self, param_dict):
        """Set params"""
        for param_name, param_value in param_dict.items():
            self.params[param_name].freeze(param_value)

    def unfreeze(self, param_names):
        """Unset params"""
        for param_name in param_names:
            self.params[param_name].unfreeze()

    def instantiate(self, sim=None):
        """Instantiate model in simulator"""
        parameters = dict((p.name, p.value) for p in self.params.values())
        parameters['v_spike'] = -40  # temporary hack; should set this elsewhere
        if self.instantiated:
            self.icell.set(**parameters)
        else:
            self.sim = sim.sim
            celltype = getattr(self.sim, self.celltype)
            self.icell = self.sim.Population(1, celltype, parameters,
                                             initial_values=self.initial_values,
                                             label=self.name)
            self.icell.record('spikes')
            self.icell.electrode = None
            self.instantiated = True

    def destroy(self):
        self.sim.reset()


class PyNNSquarePulse(ephys.stimuli.Stimulus):

    def __init__(self, step_amplitude=None, step_delay=None,
                 step_duration=None, total_duration=None):
        super(PyNNSquarePulse, self).__init__()
        self.step_amplitude = step_amplitude
        self.step_delay = step_delay
        self.step_duration = step_duration
        self.total_duration = total_duration
        self.stim = None

    def instantiate(self, sim=None, icell=None):
        parameters = dict(amplitude=self.step_amplitude,
                          start=self.step_delay,
                          stop=self.step_delay + self.step_duration)
        if icell.electrode:
            icell.electrode.set_parameters(**parameters)
        else:
            icell.electrode = sim.sim.DCSource(**parameters)
            icell.electrode.inject_into(icell)

    def destroy(self):
        pass


class PyNNRecording(ephys.recordings.Recording):
    count = 0

    def __init__(self, name=None, value=None, frozen=None, variable='v'):
        super(PyNNRecording, self).__init__(name=name, value=value, frozen=frozen)
        self.variable = variable
        self.artificial_ap = True

    def instantiate(self, sim=None, icell=None):
        """Instantiate recording"""
        self.population = icell
        if self.variable not in self.population.recorder.recorded:
            self.population.record(self.variable)
        self.instantiated = True
        self.index = self.__class__.count
        self.__class__.count += 1

    @property
    def response(self):
        """Return recording response"""
        if not self.instantiated:
            raise Exception(
                'Recording not instantiated before requesting response')
        data = self.population.get_data().segments[self.index]
        signal = data.filter(name=self.variable)[0]
        times = signal.times
        vm = signal.magnitude[:, 0]
        if self.artificial_ap:
            spike_times = data.spiketrains[0].magnitude
            if spike_times.size > 0:
                spike_indices = (spike_times/0.1).astype(int)  # dt=0.1: should get from simulator
                vm[spike_indices] = 0.0
        return ephys.responses.TimeVoltageResponse(self.name, times, vm)

    def destroy(self):
        pass


class PyNNSimulator(object):

    def __init__(self, simulator_engine):
        self.sim = import_module("pyNN.%s" % simulator_engine)
        self.sim.setup(verbosity='error')

    def run(self, tstop=None, cvode_active=True):
        """Run protocol"""
        self.sim.run(tstop)


parameters = {'tau_refrac': [0.1, 10.0], 'a': [0.0, 50.0], 'tau_m': [1.0, 50.0],
              'cm': [0.1, 10.0], 'delta_T': [0.1, 10.0], 'v_thresh': [-70.0, -41.0],
              'b': [0.0, 1.0], 'v_reset': [-80.0, -50.0], 'tau_w': [10.0, 1000.0],
              'v_rest': [-80.0, -50.0]}

parameter_objects = [ephys.parameters.Parameter(name=name, bounds=bounds, frozen=False)
                     for name, bounds in parameters.items()]


simple_cell = PyNNModel(name='simple_cell',
                        celltype='EIF_cond_exp_isfa_ista',
                        params=parameter_objects,
                        initial_values={'v': -70.0, 'w': 0.0})

sweep_protocols = []
for protocol_name, amplitude in [('step1', 0.4), ('step2', 2.0)]:
    stim = PyNNSquarePulse(
                step_amplitude=amplitude,
                step_delay=100,
                step_duration=50,
                total_duration=200)
    rec = PyNNRecording(
            name='%s.v' % protocol_name,
            variable='v')
    protocol = ephys.protocols.SweepProtocol(protocol_name, [stim], [rec])
    sweep_protocols.append(protocol)
twostep_protocol = ephys.protocols.SequenceProtocol('twostep', protocols=sweep_protocols)


simulator = PyNNSimulator('nest')

default_params = {'tau_refrac': 2.0, 'a': 4.0, 'tau_m': 10.0,
                  'cm': 0.5, 'delta_T': 2.0, 'v_thresh': -50.0,
                  'b': 0.1, 'v_reset': -70.0, 'tau_w': 100.0,
                  'v_rest': -70.0}

responses = twostep_protocol.run(cell_model=simple_cell,
                                 param_values=default_params,
                                 sim=simulator)


def plot_responses(responses):
    plt.subplot(2,1,1)
    plt.plot(responses['step1.v']['time'], responses['step1.v']['voltage'], label='step1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(responses['step2.v']['time'], responses['step2.v']['voltage'], label='step2')
    plt.legend()
    plt.tight_layout()

if PLOT_FIGURES:
    plot_responses(responses)
    plt.savefig("simple_pyNN.png")


###
import efel

def features(time, voltage, stim_limits, feature_names):
    start, end = stim_limits
    traces = [{'T': time, 'V': voltage, 'stim_start': [start], 'stim_end': [end]}]
    return efel.getFeatureValues(traces, feature_names)

print(features(responses['step1.v']['time'],
               responses['step1.v']['voltage'],
               (100.0, 150.0),
               ['Spikecount', 'voltage_base']))
print(features(responses['step2.v']['time'],
               responses['step2.v']['voltage'],
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


cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=[p.name for p in parameter_objects],
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc,
        isolate_protocols=False,
        sim=simulator)

print(cell_evaluator.evaluate_with_dicts(default_params))


# ==========

optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator,
        offspring_size=100)

final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=20)

#print('Final population: ', final_pop)

best_ind = hall_of_fame[0]
print('Best individual: ', best_ind)
print('Fitness values: ', best_ind.fitness.values)

best_ind_dict = cell_evaluator.param_dict(best_ind)
print(cell_evaluator.evaluate_with_dicts(best_ind_dict))

if PLOT_FIGURES:
    plot_responses(twostep_protocol.run(cell_model=simple_cell, param_values=best_ind_dict, sim=simulator))
    plt.savefig("simple_pyNN_best.png")


gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
for gen, fitness in zip(gen_numbers, min_fitness):
    print(gen, fitness)
if PLOT_FIGURES:
    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.xlabel('generation #')
    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    plt.ylim(0.9*min(min_fitness), 1.1 * max(min_fitness))
    plt.savefig("simple_pyNN_fitness.png")

