import datetime
import os
import random

import binutil  # required to import from dreamcoder modules

from dreamcoder.ec import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, t0, tboolean
from dreamcoder.utilities import numberOfCPUs

# Primitives
def _incr(x): return lambda x: x + 1
def _incr2(x): return lambda x: x + 2


def addN(n):
    x = random.choice(range(500))
    return {"i": x, "o": x + n}

def even():
    x = random.choice(range(500))
    return {"i": x, "o": x % 2 == 0}

def odd():
    x = random.choice(range(500))
    return {"i": x, "o": x % 2 == 1}


def get_tint_task(item):
    return Task(
        item["name"],
        arrow(tint, tboolean),
        [((ex["i"],), ex["o"]) for ex in item["examples"]],
    )


if __name__ == "__main__":

    # Options more or less copied from list.py

    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh',
        iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs())

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/demo/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create list of primitives

    primitives = [
        Primitive("+", arrow(tint, tint, tint), lambda x, y: x + y),
        Primitive("1", tint, 1),
        Primitive("2", tint, 2),
        Primitive("flip", arrow(tboolean, tboolean), lambda x: not x),
        Primitive("divide", arrow(tint, tint, tint), lambda x, y: x // y),
        Primitive("equality", arrow(t0, t0, tboolean), lambda x, y: x == y),
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    def add1(): return addN(1)
    def add2(): return addN(2)
    def add3(): return addN(3)

    # Training data

    training_examples = [
        {"name": "even", "examples": [even() for _ in range(5000)]}
    ]
    training = [get_tint_task(item) for item in training_examples]

    # Testing data

    testing_examples = [
        {"name": "odd", "examples": [odd() for _ in range(500)]}
    ]
    testing = [get_tint_task(item) for item in testing_examples]

    # EC iterate

    generator = ecIterator(grammar,
                           training,
                           testingTasks=testing,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))