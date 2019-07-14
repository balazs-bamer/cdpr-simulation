import BaseClasses as bc
import tensorflow as tf
import core


class Policy(bc.SpecializedFunction):
    input_names = ['state']
    output_names = ['action']

    def __init__(self, dtype, gs):
        super(Policy, self).__init__(dtype, gs)
