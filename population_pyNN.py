"""


"""

try:
    from importlib import import_module
except ImportError:  # Python 2.6
    def import_module(name):
        return __import__(name)
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bluepyopt as bpop
import bluepyopt.ephys as ephys


import logging
logging.basicConfig(filename='population_pyNN.log', filemode='w', level=logging.DEBUG)

PLOT_FIGURES = True
POP_SIZE = 100
N_GEN = 20


class PyNNPopulationModel(object):

    def __init__(self, name, size, celltype, params=None, initial_values=None):
        # each parameter is array-valued
        self.name = name
        self.size = size
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
            self.icell = self.sim.Population(self.size, celltype, parameters,
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
        vm = signal.magnitude
        n_signals = vm.shape[1]
        if self.artificial_ap:
            for i in range(n_signals):
                spike_times = data.spiketrains[i].magnitude
                if spike_times.size > 0:
                    spike_indices = (spike_times/0.1).astype(int)  # dt=0.1: should get from simulator
                    vm[spike_indices, i] = 0.0
        return [ephys.responses.TimeVoltageResponse(self.name, times, vm[:, i])
                for i in range(n_signals)]

    def destroy(self):
        pass


class PyNNSimulator(object):

    def __init__(self, simulator_engine):
        self.sim = import_module("pyNN.%s" % simulator_engine)
        self.sim.setup(verbosity='error')

    def run(self, tstop=None, cvode_active=True):
        """Run protocol"""
        self.sim.run(tstop)


class ArrayParameter(ephys.parameters.Parameter):

    def check_bounds(self):
        """Check if parameter is within bounds"""
        if self.bounds and self._value is not None:
            if not ((self.lower_bound <= self._value).all() and (self._value <= self.upper_bound).all()):
                raise Exception(
                    'Parameter %s has value %s outside of bounds [%s, %s]' %
                    (self.name, self._value, str(self.lower_bound),
                     str(self.upper_bound)))


class PopulationEvaluator(object):

    def __init__(
            self,
            population_model=None,
            param_names=None,
            fitness_protocols=None,
            fitness_calculator=None,
            isolate_protocols=True,
            sim=None):
        """Constructor"""

        self.population_model = population_model
        self.param_names = param_names
        # Stimuli used for fitness calculation
        self.fitness_protocols = fitness_protocols
        # Fitness value calculator
        self.fitness_calculator = fitness_calculator

        self.isolate_protocols = isolate_protocols

        self.sim = sim

    @property
    def objectives(self):
        """Return objectives"""

        return self.fitness_calculator.objectives

    @property
    def params(self):
        """Return params of this evaluation"""

        params = []
        for param_name in self.param_names:
            params.append(self.population_model.params[param_name])

        return params

    def param_dict(self, param_array):
        """Convert param_array in param_dict"""
        param_dict = {}
        param_array = np.array(param_array).transpose()
        for param_name, param_value in \
                zip(self.param_names, param_array):
            param_dict[param_name] = param_value

        return param_dict

    def objective_dict(self, objective_array):
        """Convert objective_array in objective_dict"""
        objective_dict = {}
        objective_names = [objective.name
                           for objective in self.fitness_calculator.objectives]
        for objective_name, objective_value in \
                zip(objective_names, objective_array):
            objective_dict[objective_name] = objective_value

        return objective_dict

    def run_protocol(self, protocol, param_values):
        """Run protocols"""
        return protocol.run(self.population_model, param_values, sim=self.sim)

    def run_protocols(self, protocols, param_values):
        """Run a set of protocols"""

        responses = {}

        for protocol in protocols:
            responses.update(self.run_protocol(
                protocol,
                param_values=param_values))

        return responses

    def evaluate_with_dicts(self, param_dict=None):
        """Run evaluation with dict as input and output"""

        if self.fitness_calculator is None:
            raise Exception(
                'PopulationEvaluator: need fitness_calculator to evaluate')

        responses = self.run_protocols(
            self.fitness_protocols.values(),
            param_dict)

        scores = []
        for channel in range(self.population_model.size):
            # this part should perhaps be parallelised
            response = {}
            for label in responses:
                response[label] = responses[label][channel]
            scores.append(self.fitness_calculator.calculate_scores(response))
        return scores

    def evaluate_with_lists(self, param_list=None):
        """Run evaluation with lists as input and outputs"""
        # param_list should be a list of lists, each sublist containing the parameters for one cell in the population

        param_dict = self.param_dict(param_list)

        obj_dicts = self.evaluate_with_dicts(param_dict=param_dict)

        return [obj_dict.values() for obj_dict in obj_dicts]

    def evaluate(self, param_list=None):
        """Run evaluation with lists as input and outputs"""
        return self.evaluate_with_lists(param_list)


def evaluate_fitnesses(toolbox, individuals):
    # individuals is a list of lists, each list containing the parameters for one individual
    return toolbox.evaluate(individuals)


bpop.optimisations.evaluate_fitnesses = evaluate_fitnesses




parameters = {'tau_refrac': [0.1, 10.0], 'a': [0.0, 50.0], 'tau_m': [1.0, 50.0],
              'cm': [0.1, 10.0], 'delta_T': [0.1, 10.0], 'v_thresh': [-70.0, -41.0],
              'b': [0.0, 1.0], 'v_reset': [-80.0, -50.0], 'tau_w': [10.0, 1000.0],
              'v_rest': [-80.0, -50.0]}

parameter_objects = [ArrayParameter(name=name, bounds=bounds, frozen=False)
                     for name, bounds in parameters.items()]


simple_population = PyNNPopulationModel(name='simple_population',
                                        size=POP_SIZE,
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


pop_evaluator = PopulationEvaluator(
        population_model=simple_population,
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

