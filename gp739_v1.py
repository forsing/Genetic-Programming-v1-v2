# https://github.com/guyko81/gplearn



"""
Genetic Programming 
gp739_v1 
Loto 7/39 predikcija preko SymbolicRegressor 
Samostalan embedded fajl sa scikit-learn 
"""


import itertools
from abc import ABCMeta, abstractmethod
from copy import copy
from time import time
from warnings import warn

from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import log_loss
from joblib import Parallel, delayed, cpu_count

try:
    import six
except ImportError:
    class _SixCompat:
        string_types = (str,)
        @staticmethod
        def get_function_code(function):
            return function.__code__
        @staticmethod
        def with_metaclass(mcls, base):
            return mcls("TemporaryClass", (base,), {})
    six = _SixCompat()

import numbers

import numpy as np
from joblib import cpu_count


class NotFittedError(ValueError, AttributeError):

    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling and backward compatibility.

    Examples
    --------
    >>> from sklearn.svm import LinearSVC
    >>> from sklearn.exceptions import NotFittedError
    >>> try:
    ...     LinearSVC().predict([[1, 2], [2, 3], [3, 4]])
    ... except NotFittedError as e:
    ...     print(repr(e))
    ...                        # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    NotFittedError('This LinearSVC instance is not fitted yet',)
    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.

    """


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    Examples
    --------
    >>> from sklearn.utils import _get_n_jobs
    >>> _get_n_jobs(4)
    4
    >>> jobs = _get_n_jobs(-2)
    >>> assert jobs == max(cpu_count() - 1, 1)
    >>> _get_n_jobs(0)
    Traceback (most recent call last):
    ...
    ValueError: Parameter n_jobs == 0 has no meaning.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np
try:
    import six
except ImportError:
    class _SixCompat:
        string_types = (str,)

        @staticmethod
        def get_function_code(function):
            return function.__code__

    six = _SixCompat()

__all__ = ['make_function']


class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)


def make_function(function, name, arity):
    """Make a function node, a representation of a mathematical relationship.

    This factory function creates a function node, one of the core nodes in any
    program. The resulting object is able to be called with NumPy vectorized
    arguments and return a resulting vector based on a mathematical
    relationship.

    Parameters
    ----------
    function : callable
        A function with signature `function(x1, *args)` that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the `function` takes.

    """
    if not isinstance(arity, int):
        raise ValueError('arity must be an int, got %s' % type(arity))
    if not isinstance(function, np.ufunc):
        if six.get_function_code(function).co_argcount != arity:
            raise ValueError('arity %d does not match required number of '
                             'function arguments of %d.'
                             % (arity,
                                six.get_function_code(function).co_argcount))
    if not isinstance(name, six.string_types):
        raise ValueError('name must be a string, got %s' % type(name))

    # Check output shape
    args = [np.ones(10) for _ in range(arity)]
    try:
        function(*args)
    except ValueError:
        raise ValueError('supplied function %s does not support arity of %d.'
                         % (name, arity))
    if not hasattr(function(*args), 'shape'):
        raise ValueError('supplied function %s does not return a numpy array.'
                         % name)
    if function(*args).shape != (10,):
        raise ValueError('supplied function %s does not return same shape as '
                         'input vectors.' % name)

    # Check closure for zero & negative input arguments
    args = [np.zeros(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'zeros in argument vectors.' % name)
    args = [-1 * np.ones(10) for _ in range(arity)]
    if not np.all(np.isfinite(function(*args))):
        raise ValueError('supplied function %s does not have closure against '
                         'negatives in argument vectors.' % name)

    return _Function(function, name, arity)


def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def _protected_sqrt(x1):
    """Closure of square root for negative arguments."""
    return np.sqrt(np.abs(x1))


def _protected_log(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def _protected_inverse(x1):
    """Closure of log for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


add2 = make_function(function=np.add, name='add', arity=2)
sub2 = make_function(function=np.subtract, name='sub', arity=2)
mul2 = make_function(function=np.multiply, name='mul', arity=2)
div2 = make_function(function=_protected_division, name='div', arity=2)
sqrt1 = make_function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = make_function(function=_protected_log, name='log', arity=1)
neg1 = make_function(function=np.negative, name='neg', arity=1)
inv1 = make_function(function=_protected_inverse, name='inv', arity=1)
abs1 = make_function(function=np.abs, name='abs', arity=1)
max2 = make_function(function=np.maximum, name='max', arity=2)
min2 = make_function(function=np.minimum, name='min', arity=2)
sin1 = make_function(function=np.sin, name='sin', arity=1)
cos1 = make_function(function=np.cos, name='cos', arity=1)
tan1 = make_function(function=np.tan, name='tan', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}


# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numbers

import numpy as np
from scipy.stats import rankdata
try:
    import six
except ImportError:
    class _SixCompat:
        @staticmethod
        def get_function_code(function):
            return function.__code__

    six = _SixCompat()
from sklearn.metrics import log_loss

__all__ = ['make_fitness']


class _Fitness(object):

    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def make_fitness(function, greater_is_better):
    """Make a fitness measure, a metric scoring the quality of a program's fit.

    This factory function creates a fitness measure object which measures the
    quality of a program's fit and thus its likelihood to undergo genetic
    operations into the next generation. The resulting object is able to be
    called with NumPy vectorized arguments and return a resulting floating
    point score quantifying the quality of the program's representation of the
    true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """
    if not isinstance(greater_is_better, bool):
        raise ValueError('greater_is_better must be bool, got %s'
                         % type(greater_is_better))
    if six.get_function_code(function).co_argcount != 3:
        raise ValueError('function requires 3 arguments (y, y_pred, w),'
                         ' got %d.'
                         % six.get_function_code(function).co_argcount)
    if not isinstance(function(np.array([1, 1]),
                      np.array([2, 2]),
                      np.array([1, 1])), numbers.Number):
        raise ValueError('function must return a numeric.')

    return _Fitness(function, greater_is_better)


def _weighted_pearson(y, y_pred, w):
    """Calculate the weighted Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        y_pred_demean = y_pred - np.average(y_pred, weights=w)
        y_demean = y - np.average(y, weights=w)
        corr = ((np.sum(w * y_pred_demean * y_demean) / np.sum(w)) /
                np.sqrt((np.sum(w * y_pred_demean ** 2) *
                         np.sum(w * y_demean ** 2)) /
                        (np.sum(w) ** 2)))
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


def _weighted_spearman(y, y_pred, w):
    """Calculate the weighted Spearman correlation coefficient."""
    y_pred_ranked = np.apply_along_axis(rankdata, 0, y_pred)
    y_ranked = np.apply_along_axis(rankdata, 0, y)
    return _weighted_pearson(y_pred_ranked, y_ranked, w)


def _mean_absolute_error(y, y_pred, w):
    """Calculate the mean absolute error."""
    return np.average(np.abs(y_pred - y), weights=w)


def _mean_square_error(y, y_pred, w):
    """Calculate the mean square error."""
    return np.average(((y_pred - y) ** 2), weights=w)


def _root_mean_square_error(y, y_pred, w):
    """Calculate the root mean square error."""
    return np.sqrt(np.average(((y_pred - y) ** 2), weights=w))


def _sigmoid(x):
    "Numerically stable sigmoid function."
    x = np.asarray(x)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out

def _binary_crossentropy_loss(y, y_pred, w):
    """Calculate the binary classification error."""
    y_pred_sigmoid = _sigmoid(y_pred)
    labels = np.zeros((len(y), 2))
    labels[np.arange(len(y)), y.astype(int)] = 1
    return log_loss(labels, y_pred_sigmoid, sample_weight=w)


weighted_pearson = make_fitness(function=_weighted_pearson,
                                greater_is_better=True)
weighted_spearman = make_fitness(function=_weighted_spearman,
                                 greater_is_better=True)
mean_absolute_error = make_fitness(function=_mean_absolute_error,
                                   greater_is_better=False)
mean_square_error = make_fitness(function=_mean_square_error,
                                 greater_is_better=False)
root_mean_square_error = make_fitness(function=_root_mean_square_error,
                                      greater_is_better=False)
binary_crossentropy_loss = make_fitness(function=_binary_crossentropy_loss,
                                      greater_is_better=False)

_fitness_map = {'pearson': weighted_pearson,
                'spearman': weighted_spearman,
                'mean absolute error': mean_absolute_error,
                'mse': mean_square_error,
                'rmse': root_mean_square_error,
                'binary classification': binary_crossentropy_loss}


# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

from copy import copy

import numpy as np
from sklearn.utils.random import sample_without_replacement




class _Program(object):

    """A program-like representation of the evolved program.

    This is the underlying data-structure used by the public classes in this
    module. It should not be used directly by the user.

    Parameters
    ----------
    function_set : list
        A list of valid functions to use in the program.

    arities : dict
        A dictionary of the form `{arity: [functions]}`. The arity is the
        number of arguments that the function takes, the functions must match
        those in the `function_set` parameter.

    init_depth : tuple of two ints
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    n_features : int
        The number of features in `X`.

    const_range : tuple of two floats
        The range of constants to include in the formulas.

    metric : _Fitness object
        The raw fitness metric.

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.

    parsimony_coefficient : float
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

    random_state : RandomState instance
        The random number generator. Note that ints, or None are not allowed.
        The reason for this being passed is that during parallel evolution the
        same program object may be accessed by multiple parallel processes.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.

    raw_fitness_ : float
        The raw fitness of the individual program.

    fitness_ : float
        The penalized fitness of the individual program.

    oob_fitness_ : float
        The out-of-bag raw fitness of the individual program for the held-out
        samples. Only present when sub-sampling was used in the estimator by
        specifying `max_samples` < 1.0.

    parents : dict, or None
        If None, this is a naive random program from the initial population.
        Otherwise it includes meta-data about the program's parent(s) as well
        as the genetic operations performed to yield the current program. This
        is set outside this class by the controlling evolution loops.

    depth_ : int
        The maximum depth of the program tree.

    length_ : int
        The number of functions and terminals in the program.

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 feature_names=None,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.feature_names = feature_names
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None

    def build_program(self, random_state):
        """Build a naive random program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        function = random_state.randint(len(self.function_set))
        function = self.function_set[function]
        program = [function]
        terminal_stack = [function.arity]

        while terminal_stack:
            depth = len(terminal_stack)
            choice = self.n_features + len(self.function_set)
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = self.function_set[function]
                program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                while terminal_stack[-1] == 0:
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
                    output += ')'
                if i != len(self.program) - 1:
                    output += ', '
        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                             else X[:, t] if isinstance(t, int)
                             else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    def raw_fitness(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute(X)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness

    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return copy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = copy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)


MAX_INT = np.iinfo(np.int32).max


def _parallel_evolve(n_programs, parents, X, y, sample_weight, seeds, params):
    """Private function used to build a batch of programs within a job."""
    n_samples, n_features = X.shape
    # Unpack parameters
    tournament_size = params['tournament_size']
    function_set = params['function_set']
    arities = params['arities']
    init_depth = params['init_depth']
    init_method = params['init_method']
    const_range = params['const_range']
    metric = params['_metric']
    parsimony_coefficient = params['parsimony_coefficient']
    method_probs = params['method_probs']
    p_point_replace = params['p_point_replace']
    max_samples = params['max_samples']
    feature_names = params['feature_names']

    max_samples = int(max_samples * n_samples)

    def _tournament():
        """Find the fittest individual from a sub-population."""
        contenders = random_state.randint(0, len(parents), tournament_size)
        fitness = [parents[p].fitness_ for p in contenders]
        if metric.greater_is_better:
            parent_index = contenders[np.argmax(fitness)]
        else:
            parent_index = contenders[np.argmin(fitness)]
        return parents[parent_index], parent_index

    # Build programs
    programs = []

    for i in range(n_programs):

        random_state = check_random_state(seeds[i])

        if parents is None:
            program = None
            genome = None
        else:
            method = random_state.uniform()
            parent, parent_index = _tournament()

            if method < method_probs[0]:
                # crossover
                donor, donor_index = _tournament()
                program, removed, remains = parent.crossover(donor.program,
                                                             random_state)
                genome = {'method': 'Crossover',
                          'parent_idx': parent_index,
                          'parent_nodes': removed,
                          'donor_idx': donor_index,
                          'donor_nodes': remains}
            elif method < method_probs[1]:
                # subtree_mutation
                program, removed, _ = parent.subtree_mutation(random_state)
                genome = {'method': 'Subtree Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[2]:
                # hoist_mutation
                program, removed = parent.hoist_mutation(random_state)
                genome = {'method': 'Hoist Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': removed}
            elif method < method_probs[3]:
                # point_mutation
                program, mutated = parent.point_mutation(random_state)
                genome = {'method': 'Point Mutation',
                          'parent_idx': parent_index,
                          'parent_nodes': mutated}
            else:
                # reproduction
                program = parent.reproduce()
                genome = {'method': 'Reproduction',
                          'parent_idx': parent_index,
                          'parent_nodes': []}

        program = _Program(function_set=function_set,
                           arities=arities,
                           init_depth=init_depth,
                           init_method=init_method,
                           n_features=n_features,
                           metric=metric,
                           const_range=const_range,
                           p_point_replace=p_point_replace,
                           parsimony_coefficient=parsimony_coefficient,
                           feature_names=feature_names,
                           random_state=random_state,
                           program=program)

        program.parents = genome

        # Draw samples, using sample weights, and then fit
        if sample_weight is None:
            curr_sample_weight = np.ones((n_samples,))
        else:
            curr_sample_weight = sample_weight.copy()
        oob_sample_weight = curr_sample_weight.copy()

        indices, not_indices = program.get_all_indices(n_samples,
                                                       max_samples,
                                                       random_state)

        curr_sample_weight[not_indices] = 0
        oob_sample_weight[indices] = 0

        program.raw_fitness_ = program.raw_fitness(X, y, curr_sample_weight)
        if max_samples < n_samples:
            # Calculate OOB fitness
            program.oob_fitness_ = program.raw_fitness(X, y, oob_sample_weight)

        programs.append(program)

    return programs


class BaseSymbolic(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Base class for symbolic regression / classification estimators.

    Warning: This class should not be used directly.
    Use derived classes instead.

    """

    @abstractmethod
    def __init__(self,
                 population_size=1000,
                 hall_of_fame=None,
                 n_components=None,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):

        self.population_size = population_size
        self.hall_of_fame = hall_of_fame
        self.n_components = n_components
        self.generations = generations
        self.tournament_size = tournament_size
        self.stopping_criteria = stopping_criteria
        self.const_range = const_range
        self.init_depth = init_depth
        self.init_method = init_method
        self.function_set = function_set
        self.metric = metric
        self.parsimony_coefficient = parsimony_coefficient
        self.p_crossover = p_crossover
        self.p_subtree_mutation = p_subtree_mutation
        self.p_hoist_mutation = p_hoist_mutation
        self.p_point_mutation = p_point_mutation
        self.p_point_replace = p_point_replace
        self.max_samples = max_samples
        self.feature_names = feature_names
        self.warm_start = warm_start
        self.low_memory = low_memory
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _verbose_reporter(self, run_details=None):
        """A report of the progress of the evolution process.

        Parameters
        ----------
        run_details : dict
            Information about the evolution.

        """
        if run_details is None:
            print('    |{:^25}|{:^42}|'.format('Population Average',
                                               'Best Individual'))
            print('-' * 4 + ' ' + '-' * 25 + ' ' + '-' * 42 + ' ' + '-' * 10)
            line_format = '{:>4} {:>8} {:>16} {:>8} {:>16} {:>16} {:>10}'
            print(line_format.format('Gen', 'Length', 'Fitness', 'Length',
                                     'Fitness', 'OOB Fitness', 'Time Left'))

        else:
            # Estimate remaining time for run
            gen = run_details['generation'][-1]
            generation_time = run_details['generation_time'][-1]
            remaining_time = (self.generations - gen - 1) * generation_time
            if remaining_time > 60:
                remaining_time = '{0:.2f}m'.format(remaining_time / 60.0)
            else:
                remaining_time = '{0:.2f}s'.format(remaining_time)

            oob_fitness = 'N/A'
            line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:>16} {:>10}'
            if self.max_samples < 1.0:
                oob_fitness = run_details['best_oob_fitness'][-1]
                line_format = '{:4d} {:8.2f} {:16g} {:8d} {:16g} {:16g} {:>10}'

            print(line_format.format(run_details['generation'][-1],
                                     run_details['average_length'][-1],
                                     run_details['average_fitness'][-1],
                                     run_details['best_length'][-1],
                                     run_details['best_fitness'][-1],
                                     oob_fitness,
                                     remaining_time))

    def fit(self, X, y, sample_weight=None):
        """Fit the Genetic Program according to X, y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples.

        Returns
        -------
        self : object
            Returns self.

        """
        random_state = check_random_state(self.random_state)

        # Check arrays
        X, y = check_X_y(X, y, y_numeric=True)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        _, self.n_features_ = X.shape

        hall_of_fame = self.hall_of_fame
        if hall_of_fame is None:
            hall_of_fame = self.population_size
        if hall_of_fame > self.population_size or hall_of_fame < 1:
            raise ValueError('hall_of_fame (%d) must be less than or equal to '
                             'population_size (%d).' % (self.hall_of_fame,
                                                        self.population_size))
        n_components = self.n_components
        if n_components is None:
            n_components = hall_of_fame
        if n_components > hall_of_fame or n_components < 1:
            raise ValueError('n_components (%d) must be less than or equal to '
                             'hall_of_fame (%d).' % (self.n_components,
                                                     self.hall_of_fame))

        self._function_set = []
        for function in self.function_set:
            if isinstance(function, six.string_types):
                if function not in _function_map:
                    raise ValueError('invalid function name %s found in '
                                     '`function_set`.' % function)
                self._function_set.append(_function_map[function])
            elif isinstance(function, _Function):
                self._function_set.append(function)
            else:
                raise ValueError('invalid type %s found in `function_set`.'
                                 % type(function))
        if not self._function_set:
            raise ValueError('No valid functions found in `function_set`.')

        # For point-mutation to find a compatible replacement node
        self._arities = {}
        for function in self._function_set:
            arity = function.arity
            self._arities[arity] = self._arities.get(arity, [])
            self._arities[arity].append(function)

        if isinstance(self.metric, _Fitness):
            self._metric = self.metric
        elif isinstance(self, RegressorMixin):
            if self.metric not in ('mean absolute error', 'mse', 'rmse',
                                   'pearson', 'spearman', 'binary classification'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            else:
                self._metric = _fitness_map[self.metric]
        elif isinstance(self, TransformerMixin):
            if self.metric not in ('pearson', 'spearman'):
                raise ValueError('Unsupported metric: %s' % self.metric)
            else:
                self._metric = _fitness_map[self.metric]

        self._method_probs = np.array([self.p_crossover,
                                       self.p_subtree_mutation,
                                       self.p_hoist_mutation,
                                       self.p_point_mutation])
        self._method_probs = np.cumsum(self._method_probs)

        if self._method_probs[-1] > 1:
            raise ValueError('The sum of p_crossover, p_subtree_mutation, '
                             'p_hoist_mutation and p_point_mutation should '
                             'total to 1.0 or less.')

        if self.init_method not in ('half and half', 'grow', 'full'):
            raise ValueError('Valid program initializations methods include '
                             '"grow", "full" and "half and half". Given %s.'
                             % self.init_method)

        if not((isinstance(self.const_range, tuple) and
                len(self.const_range) == 2) or self.const_range is None):
            raise ValueError('const_range should be a tuple with length two, '
                             'or None.')

        if (not isinstance(self.init_depth, tuple) or
                len(self.init_depth) != 2):
            raise ValueError('init_depth should be a tuple with length two.')
        if self.init_depth[0] > self.init_depth[1]:
            raise ValueError('init_depth should be in increasing numerical '
                             'order: (min_depth, max_depth).')

        if self.feature_names is not None:
            if self.n_features_ != len(self.feature_names):
                raise ValueError('The supplied `feature_names` has different '
                                 'length to n_features. Expected %d, got %d.'
                                 % (self.n_features_, len(self.feature_names)))
            for feature_name in self.feature_names:
                if not isinstance(feature_name, six.string_types):
                    raise ValueError('invalid type %s found in '
                                     '`feature_names`.' % type(feature_name))

        params = self.get_params()
        params['_metric'] = self._metric
        params['function_set'] = self._function_set
        params['arities'] = self._arities
        params['method_probs'] = self._method_probs

        if not self.warm_start or not hasattr(self, '_programs'):
            # Free allocated memory, if any
            self._programs = []
            self.run_details_ = {'generation': [],
                                 'average_length': [],
                                 'average_fitness': [],
                                 'best_length': [],
                                 'best_fitness': [],
                                 'best_oob_fitness': [],
                                 'generation_time': []}

        prior_generations = len(self._programs)
        n_more_generations = self.generations - prior_generations

        if n_more_generations < 0:
            raise ValueError('generations=%d must be larger or equal to '
                             'len(_programs)=%d when warm_start==True'
                             % (self.generations, len(self._programs)))
        elif n_more_generations == 0:
            fitness = [program.raw_fitness_ for program in self._programs[-1]]
            warn('Warm-start fitting without increasing n_estimators does not '
                 'fit new programs.')

        if self.warm_start:
            # Generate and discard seeds that would have been produced on the
            # initial fit call.
            for i in range(len(self._programs)):
                _ = random_state.randint(MAX_INT, size=self.population_size)

        if self.verbose:
            # Print header fields
            self._verbose_reporter()

        for gen in range(prior_generations, self.generations):

            start_time = time()

            if gen == 0:
                parents = None
            else:
                parents = self._programs[gen - 1]

            # Parallel loop
            n_jobs, n_programs, starts = _partition_estimators(
                self.population_size, self.n_jobs)
            seeds = random_state.randint(MAX_INT, size=self.population_size)

            population = Parallel(n_jobs=n_jobs,
                                  verbose=int(self.verbose > 1))(
                delayed(_parallel_evolve)(n_programs[i],
                                          parents,
                                          X,
                                          y,
                                          sample_weight,
                                          seeds[starts[i]:starts[i + 1]],
                                          params)
                for i in range(n_jobs))

            # Reduce, maintaining order across different n_jobs
            population = list(itertools.chain.from_iterable(population))

            fitness = [program.raw_fitness_ for program in population]
            length = [program.length_ for program in population]

            parsimony_coefficient = None
            if self.parsimony_coefficient == 'auto':
                parsimony_coefficient = (np.cov(length, fitness)[1, 0] /
                                         np.var(length))
            for program in population:
                program.fitness_ = program.fitness(parsimony_coefficient)

            self._programs.append(population)

            # Remove old programs that didn't make it into the new population.
            if not self.low_memory:
                for old_gen in np.arange(gen, 0, -1):
                    indices = []
                    for program in self._programs[old_gen]:
                        if program is not None:
                            for idx in program.parents:
                                if 'idx' in idx:
                                    indices.append(program.parents[idx])
                    indices = set(indices)
                    for idx in range(self.population_size):
                        if idx not in indices:
                            self._programs[old_gen - 1][idx] = None
            elif gen > 0:
                # Remove old generations
                self._programs[gen - 1] = None

            # Record run details
            if self._metric.greater_is_better:
                best_program = population[np.argmax(fitness)]
            else:
                best_program = population[np.argmin(fitness)]

            self.run_details_['generation'].append(gen)
            self.run_details_['average_length'].append(np.mean(length))
            self.run_details_['average_fitness'].append(np.mean(fitness))
            self.run_details_['best_length'].append(best_program.length_)
            self.run_details_['best_fitness'].append(best_program.raw_fitness_)
            oob_fitness = np.nan
            if self.max_samples < 1.0:
                oob_fitness = best_program.oob_fitness_
            self.run_details_['best_oob_fitness'].append(oob_fitness)
            generation_time = time() - start_time
            self.run_details_['generation_time'].append(generation_time)

            if self.verbose:
                self._verbose_reporter(self.run_details_)

            # Check for early stopping
            if self._metric.greater_is_better:
                best_fitness = fitness[np.argmax(fitness)]
                if best_fitness >= self.stopping_criteria:
                    break
            else:
                best_fitness = fitness[np.argmin(fitness)]
                if best_fitness <= self.stopping_criteria:
                    break

        if isinstance(self, RegressorMixin):
            # Find the best individual in the final generation
            if self._metric.greater_is_better:
                self._program = self._programs[-1][np.argmax(fitness)]
            else:
                self._program = self._programs[-1][np.argmin(fitness)]

        if isinstance(self, TransformerMixin):
            # Find the best individuals in the final generation
            fitness = np.array(fitness)
            if self._metric.greater_is_better:
                hall_of_fame = fitness.argsort()[::-1][:self.hall_of_fame]
            else:
                hall_of_fame = fitness.argsort()[:self.hall_of_fame]
            evaluation = np.array([gp.execute(X) for gp in
                                   [self._programs[-1][i] for
                                    i in hall_of_fame]])
            if self.metric == 'spearman':
                evaluation = np.apply_along_axis(rankdata, 1, evaluation)

            with np.errstate(divide='ignore', invalid='ignore'):
                correlations = np.abs(np.corrcoef(evaluation))
            np.fill_diagonal(correlations, 0.)
            components = list(range(self.hall_of_fame))
            indices = list(range(self.hall_of_fame))
            # Iteratively remove least fit individual of most correlated pair
            while len(components) > self.n_components:
                most_correlated = np.unravel_index(np.argmax(correlations),
                                                   correlations.shape)
                # The correlation matrix is sorted by fitness, so identifying
                # the least fit of the pair is simply getting the higher index
                worst = max(most_correlated)
                components.pop(worst)
                indices.remove(worst)
                correlations = correlations[:, indices][indices, :]
                indices = list(range(len(components)))
            self._best_programs = [self._programs[-1][i] for i in
                                   hall_of_fame[components]]

        return self


class SymbolicRegressor(BaseSymbolic, RegressorMixin):

    """A Genetic Programming symbolic regressor.

    A symbolic regressor is an estimator that begins by building a population
    of naive random formulas to represent a relationship. The formulas are
    represented as tree-like structures with mathematical functions being
    recursively applied to variables and constants. Each successive generation
    of programs is then evolved from the one that came before it by selecting
    the fittest individuals from the population to undergo genetic operations
    such as crossover, mutation or reproduction.

    Parameters
    ----------
    population_size : integer, optional (default=1000)
        The number of programs in each generation.

    generations : integer, optional (default=20)
        The number of generations to evolve.

    tournament_size : integer, optional (default=20)
        The number of programs that will compete to become part of the next
        generation.

    stopping_criteria : float, optional (default=0.0)
        The required metric value required in order to stop evolution early.

    const_range : tuple of two floats, or None, optional (default=(-1., 1.))
        The range of constants to include in the formulas. If None then no
        constants will be included in the candidate programs.

    init_depth : tuple of two ints, optional (default=(2, 6))
        The range of tree depths for the initial population of naive formulas.
        Individual trees will randomly choose a maximum depth from this range.
        When combined with `init_method='half and half'` this yields the well-
        known 'ramped half and half' initialization method.

    init_method : str, optional (default='half and half')
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
        - 'half and half' : Trees are grown through a 50/50 mix of 'full' and
          'grow', making for a mix of tree shapes in the initial population.

    function_set : iterable, optional (default=('add', 'sub', 'mul', 'div'))
        The functions to use when building and evolving programs. This iterable
        can include strings to indicate either individual functions as outlined
        below, or you can also include your own functions as built using the
        ``make_function`` factory from the ``functions`` module.

        Available individual functions are:

        - 'add' : addition, arity=2.
        - 'sub' : subtraction, arity=2.
        - 'mul' : multiplication, arity=2.
        - 'div' : protected division where a denominator near-zero returns 1.,
          arity=2.
        - 'sqrt' : protected square root where the absolute value of the
          argument is used, arity=1.
        - 'log' : protected log where the absolute value of the argument is
          used and a near-zero argument returns 0., arity=1.
        - 'abs' : absolute value, arity=1.
        - 'neg' : negative, arity=1.
        - 'inv' : protected inverse where a near-zero argument returns 0.,
          arity=1.
        - 'max' : maximum, arity=2.
        - 'min' : minimum, arity=2.
        - 'sin' : sine (radians), arity=1.
        - 'cos' : cosine (radians), arity=1.
        - 'tan' : tangent (radians), arity=1.

    metric : str, optional (default='mean absolute error')
        The name of the raw fitness metric. Available options include:

        - 'mean absolute error'.
        - 'mse' for mean squared error.
        - 'rmse' for root mean squared error.
        - 'pearson', for Pearson's product-moment correlation coefficient.
        - 'spearman' for Spearman's rank-order correlation coefficient.

        Note that 'pearson' and 'spearman' will not directly predict the target
        but could be useful as value-added features in a second-step estimator.
        This would allow the user to generate one engineered feature at a time,
        using the SymbolicTransformer would allow creation of multiple features
        at once.

    parsimony_coefficient : float or "auto", optional (default=0.001)
        This constant penalizes large programs by adjusting their fitness to
        be less favorable for selection. Larger values penalize the program
        more which can control the phenomenon known as 'bloat'. Bloat is when
        evolution is increasing the size of programs without a significant
        increase in fitness, which is costly for computation time and makes for
        a less understandable final result. This parameter may need to be tuned
        over successive runs.

        If "auto" the parsimony coefficient is recalculated for each generation
        using c = Cov(l,f)/Var( l), where Cov(l,f) is the covariance between
        program size l and program fitness f in the population, and Var(l) is
        the variance of program sizes.

    p_crossover : float, optional (default=0.9)
        The probability of performing crossover on a tournament winner.
        Crossover takes the winner of a tournament and selects a random subtree
        from it to be replaced. A second tournament is performed to find a
        donor. The donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring in the next
        generation.

    p_subtree_mutation : float, optional (default=0.01)
        The probability of performing subtree mutation on a tournament winner.
        Subtree mutation takes the winner of a tournament and selects a random
        subtree from it to be replaced. A donor subtree is generated at random
        and this is inserted into the original parent to form an offspring in
        the next generation.

    p_hoist_mutation : float, optional (default=0.01)
        The probability of performing hoist mutation on a tournament winner.
        Hoist mutation takes the winner of a tournament and selects a random
        subtree from it. A random subtree of that subtree is then selected
        and this is 'hoisted' into the original subtrees location to form an
        offspring in the next generation. This method helps to control bloat.

    p_point_mutation : float, optional (default=0.01)
        The probability of performing point mutation on a tournament winner.
        Point mutation takes the winner of a tournament and selects random
        nodes from it to be replaced. Terminals are replaced by other terminals
        and functions are replaced by other functions that require the same
        number of arguments as the original node. The resulting tree forms an
        offspring in the next generation.

        Note : The above genetic operation probabilities must sum to less than
        one. The balance of probability is assigned to 'reproduction', where a
        tournament winner is cloned and enters the next generation unmodified.

    p_point_replace : float, optional (default=0.05)
        For point mutation only, the probability that any given node will be
        mutated.

    max_samples : float, optional (default=1.0)
        The fraction of samples to draw from X to evaluate each program on.

    feature_names : list, optional (default=None)
        Optional list of feature names, used purely for representations in
        the `print` operation or `export_graphviz`. If None, then X0, X1, etc
        will be used for representations.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more generations to the evolution, otherwise, just fit a new
        evolution.

    low_memory : bool, optional (default=False)
        When set to ``True``, only the current generation is retained. Parent
        information is discarded. For very large populations or runs with many
        generations, this can result in substantial memory use reduction.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for `fit`. If -1, then the number
        of jobs is set to the number of cores.

    verbose : int, optional (default=0)
        Controls the verbosity of the evolution building process.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    run_details_ : dict
        Details of the evolution process. Includes the following elements:

        - 'generation' : The generation index.
        - 'average_length' : The average program length of the generation.
        - 'average_fitness' : The average program fitness of the generation.
        - 'best_length' : The length of the best program in the generation.
        - 'best_fitness' : The fitness of the best program in the generation.
        - 'best_oob_fitness' : The out of bag fitness of the best program in
          the generation (requires `max_samples` < 1.0).
        - 'generation_time' : The time it took for the generation to evolve.

    See Also
    --------
    SymbolicTransformer

    References
    ----------
    .. [1] J. Koza, "Genetic Programming", 1992.

    .. [2] R. Poli, et al. "A Field Guide to Genetic Programming", 2008.

    """

    def __init__(self,
                 population_size=1000,
                 generations=20,
                 tournament_size=20,
                 stopping_criteria=0.0,
                 const_range=(-1., 1.),
                 init_depth=(2, 6),
                 init_method='half and half',
                 function_set=('add', 'sub', 'mul', 'div'),
                 metric='mean absolute error',
                 parsimony_coefficient=0.001,
                 p_crossover=0.9,
                 p_subtree_mutation=0.01,
                 p_hoist_mutation=0.01,
                 p_point_mutation=0.01,
                 p_point_replace=0.05,
                 max_samples=1.0,
                 feature_names=None,
                 warm_start=False,
                 low_memory=False,
                 n_jobs=1,
                 verbose=0,
                 random_state=None):
        super(SymbolicRegressor, self).__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=stopping_criteria,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            feature_names=feature_names,
            warm_start=warm_start,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state)

    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        if not hasattr(self, '_program'):
            return self.__repr__()
        return self._program.__str__()

    def predict(self, X):
        """Perform regression on test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input vectors, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array, shape = [n_samples]
            Predicted values for X.

        """
        if not hasattr(self, '_program'):
            raise NotFittedError('SymbolicRegressor not fitted.')

        X = check_array(X)
        _, n_features = X.shape
        if self.n_features_ != n_features:
            raise ValueError('Number of features of the model must match the '
                             'input. Model n_features is %s and input '
                             'n_features is %s.'
                             % (self.n_features_, n_features))

        y = self._program.execute(X)

        return y



import numpy as np
import pandas as pd



CSV_PATH = "/data/loto7hh_4592_k27.csv"
COLS = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6", "Num7"]
SEED = 39
np.random.seed(SEED)

MIN_POS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=int)
MAX_POS = np.array([33, 34, 35, 36, 37, 38, 39], dtype=int)


def load_draws(path):
    df = pd.read_csv(path)
    if all(c in df.columns for c in COLS):
        return df[COLS].values.astype(float)
    return pd.read_csv(path, header=None).iloc[:, :7].values.astype(float)


def enforce_loto_7_39(nums):
    nums = np.rint(np.asarray(nums, dtype=float)).astype(int)
    nums = np.clip(nums, MIN_POS, MAX_POS)
    nums = np.sort(nums)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    for i in range(6, -1, -1):
        high = MAX_POS[i] if i == 6 else min(MAX_POS[i], nums[i + 1] - 1)
        nums[i] = min(nums[i], high)

    for i in range(7):
        low = MIN_POS[i] if i == 0 else max(MIN_POS[i], nums[i - 1] + 1)
        nums[i] = max(nums[i], low)

    return nums


def build_model(random_state):
    return SymbolicRegressor(random_state=random_state, n_jobs=1, verbose=0, **MODEL_CONFIGS[0])


MODEL_CONFIGS = [
    dict(
        population_size=1600,
        generations=24,
        tournament_size=24,
        stopping_criteria=0.0,
        const_range=(-3.0, 3.0),
        init_depth=(2, 7),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "max", "min"),
        metric="mse",
        parsimony_coefficient=0.0002,
        p_crossover=0.88,
        p_subtree_mutation=0.03,
        p_hoist_mutation=0.01,
        p_point_mutation=0.03,
        p_point_replace=0.08,
        max_samples=0.92,
    ),
    dict(
        population_size=1300,
        generations=20,
        tournament_size=20,
        stopping_criteria=0.0,
        const_range=(-2.0, 2.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "max", "min"),
        metric="mse",
        parsimony_coefficient=0.0005,
        p_crossover=0.90,
        p_subtree_mutation=0.02,
        p_hoist_mutation=0.01,
        p_point_mutation=0.02,
        p_point_replace=0.05,
        max_samples=0.95,
    ),
    dict(
        population_size=1100,
        generations=18,
        tournament_size=18,
        stopping_criteria=0.0,
        const_range=(-1.5, 1.5),
        init_depth=(2, 5),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "max", "min"),
        metric="mse",
        parsimony_coefficient=0.0010,
        p_crossover=0.92,
        p_subtree_mutation=0.015,
        p_hoist_mutation=0.01,
        p_point_mutation=0.015,
        p_point_replace=0.04,
        max_samples=0.98,
    ),
]


def build_model_from_config(cfg, random_state):
    return SymbolicRegressor(random_state=random_state, n_jobs=1, verbose=0, **cfg)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def main():
    draws = load_draws(CSV_PATH)
    X = draws[:-1]
    Y = draws[1:]
    X_next = draws[-1:].copy()

    preds = []
    print("=" * 72)
    print("gp739_v1 - Loto 7/39")
    print("=" * 72)
    print(f"CSV: {CSV_PATH}")
    print(f"Uzoraka za trening: {len(X)}")
    print()

    val_size = min(450, max(200, len(X) // 10))
    X_train, X_val = X[:-val_size], X[-val_size:]

    for pos in range(7):
        y = Y[:, pos]
        y_train, y_val = y[:-val_size], y[-val_size:]

        candidates = []
        print(f"[pozicija {pos + 1}]")
        for cfg_idx, cfg in enumerate(MODEL_CONFIGS):
            for rep in range(2):
                rs = SEED + pos * 20 + cfg_idx * 5 + rep
                model = build_model_from_config(cfg, rs)
                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                score = mae(y_val, val_pred)
                next_pred = float(np.asarray(model.predict(X_next)).ravel()[0])
                candidates.append((score, next_pred))
                print(f"  cfg{cfg_idx + 1}/r{rep + 1}: mae={score:.4f}, next={next_pred:.6f}")

        candidates.sort(key=lambda x: x[0])
        best_next = [candidates[0][1], candidates[1][1], candidates[2][1]]
        p = float(np.median(best_next))
        preds.append(p)
        print(f"  => best-3 median: {p:.6f}")

    final_pred = enforce_loto_7_39(preds)
    print()
    print("Raw ensemble (pre enforce):", np.round(np.asarray(preds, dtype=float), 6))
    print("Predicted next loto 7/39 combination:", final_pred)
    print("=" * 72)


if __name__ == "__main__":
    main()



################################################################################
################################################################################



"""
optimizovani parametri 
i dodat stabilniji trening 
(validacija + ansambl po seed-ovima) 
da poboljša kvalitet predikcije

konkretna poboljšanja: 
time-series validaciju, 
više jačih konfiguracija 
i ansambl (median) najboljih modela po poziciji. 
To daje stabilniju predikciju od jednog modela.

više konfiguracija GP modela (MODEL_CONFIGS) umesto jedne fiksne
time-series validaciju (zadnji deo kao val)
po poziciji ima 3 konfiguracije × 2 rep = 6 kandidata.
za svaku poziciju trenira više kandidata (3 cfg x 2 rep = 6)
računa MAE na validaciji
za finalnu vrednost pozicije uzima median najboljih 3 kandidata
zadržan SEED=39 i validacija izlaza u 7/39 kombinaciju
Efekat:
stabilniji rezultat
manji rizik da jedna loša random inicijalizacija pokvari predikciju
više "robustan" izlaz za loto niz

ispis raw ansambl (pre enforce) vrednosti 
i finalnog enforce koraka u kombinaciju
tačno se vidi kako se formira kombinacija
"""



################################################################################
################################################################################



"""
========================================================================
gp739_v1 - Loto 7/39
========================================================================
CSV: /data/loto7hh_4592_k27.csv
Uzoraka za trening: 4591

[pozicija 1]
  cfg1/r1: mae=3.0399, next=5.068487
  cfg1/r2: mae=3.1019, next=5.428530
  cfg2/r1: mae=3.0909, next=5.351287
  cfg2/r2: mae=3.0671, next=5.272188
  cfg3/r1: mae=3.0701, next=5.275468
  cfg3/r2: mae=3.0557, next=5.123291
  => best-3 median: 5.123291
[pozicija 2]
  cfg1/r1: mae=4.4008, next=10.314811
  cfg1/r2: mae=4.4053, next=10.317897
  cfg2/r1: mae=4.4020, next=10.295984
  cfg2/r2: mae=4.4110, next=10.140324
  cfg3/r1: mae=4.4049, next=10.256761
  cfg3/r2: mae=4.3902, next=10.242980
  => best-3 median: 10.295984
[pozicija 3]
  cfg1/r1: mae=4.9933, next=15.564766
  cfg1/r2: mae=4.9987, next=15.378589
  cfg2/r1: mae=4.9880, next=x
  cfg2/r2: mae=5.0123, next=x
  cfg3/r1: mae=4.9910, next=15.173961
  cfg3/r2: mae=4.9982, next=15.358554
  => best-3 median: x
[pozicija 4]
  cfg1/r1: mae=5.2814, next=21.137863
  cfg1/r2: mae=5.2480, next=20.614205
  cfg2/r1: mae=5.2464, next=y
  cfg2/r2: mae=5.2548, next=y
  cfg3/r1: mae=5.3814, next=21.453165
  cfg3/r2: mae=5.2967, next=21.102050
  => best-3 median: y
[pozicija 5]
  cfg1/r1: mae=4.7095, next=25.090528
  cfg1/r2: mae=4.9037, next=25.433025
  cfg2/r1: mae=4.8184, next=z
  cfg2/r2: mae=4.8043, next=z
  cfg3/r1: mae=4.9303, next=26.219732
  cfg3/r2: mae=5.1294, next=27.174525
  => best-3 median: z
[pozicija 6]
  cfg1/r1: mae=4.1878, next=31.029854
  cfg1/r2: mae=4.7387, next=32.891098
  cfg2/r1: mae=4.5037, next=31.000906
  cfg2/r2: mae=4.2387, next=31.419188
  cfg3/r1: mae=4.7498, next=32.968403
  cfg3/r2: mae=4.7542, next=32.754742
  => best-3 median: 31.029854
[pozicija 7]
  cfg1/r1: mae=3.1225, next=36.362331
  cfg1/r2: mae=3.4735, next=38.162221
  cfg2/r1: mae=3.7778, next=38.000000
  cfg2/r2: mae=3.7778, next=38.000000
  cfg3/r1: mae=3.7778, next=38.000000
  cfg3/r2: mae=3.7842, next=37.361904
  => best-3 median: 38.000000

Raw ensemble (pre enforce): 
[ 5.123291 10.295984 x y z 31.029854 38.      ]
Predicted next loto 7/39 combination: 
[ 5 10 x y z 31 38]
========================================================================
"""
