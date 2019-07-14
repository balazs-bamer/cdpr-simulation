import BaseClasses as bc


class RNN(bc.GraphStructure):

    rnn_cell_from_tensorflow = ['GRUCell', 'LSTMCell']
    custom_rnn_cell = ['GRUPartialCell']

    def __init__(self, dtype):
        super(RNN, self).__init__(dtype)
        pass


