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
        if self.instantiated:
            self.icell.set(**parameters)
        else:
            self.sim = sim.sim
            celltype = getattr(self.sim, self.celltype)
            self.icell = self.sim.Population(1, celltype, parameters,
                                             initial_values=self.initial_values,
                                             label=self.name)
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
        return ephys.responses.TimeVoltageResponse(self.name,
                                                   signal.times,
                                                   signal.magnitude[:, 0])

    def destroy(self):
        pass


class PyNNSimulator(object):

    def __init__(self, simulator_engine):
        self.sim = import_module("pyNN.%s" % simulator_engine)
        self.sim.setup()

    def run(self, tstop=None, cvode_active=True):
        """Run protocol"""
        self.sim.run(tstop)


parameters = {'tau_refrac': [0.1, 10.0], 'a': [0.0, 50.0], 'tau_m': [1.0, 50.0],
              'cm': [0.1, 10.0], 'delta_T': [0.1, 10.0], 'v_thresh': [-70.0, -41.0],
              'b': [0.0, 1.0], 'v_reset': [-80.0, -50.0], 'tau_w': [10.0, 1000.0],
              'v_rest': [-80.0, -50.0]}

parameter_objects = [ephys.parameters.Parameter(name=name, bounds=bounds, frozen=False)
                     for name, bounds in parameters.items()]
parameter_objects.append(ephys.parameters.Parameter(name='v_spike', value=-40.0,
                                                    bounds=None, frozen=True))


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


nrn = PyNNSimulator('nest')

default_params = {'tau_refrac': 2.0, 'a': 4.0, 'tau_m': 10.0,
                  'cm': 0.5, 'delta_T': 2.0, 'v_thresh': -50.0,
                  'b': 0.1, 'v_reset': -70.0, 'tau_w': 100.0,
                  'v_rest': -70.0}

responses = twostep_protocol.run(cell_model=simple_cell, param_values=default_params, sim=nrn)


def plot_responses(responses):
    plt.subplot(2,1,1)
    plt.plot(responses['step1.v']['time'], responses['step1.v']['voltage'], label='step1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(responses['step2.v']['time'], responses['step2.v']['voltage'], label='step2')
    plt.legend()
    plt.tight_layout()

plot_responses(responses)
plt.savefig("simple_pyNN.png")
