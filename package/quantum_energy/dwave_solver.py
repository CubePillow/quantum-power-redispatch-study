import pickle

import dimod
import numpy as np
import qubolite
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite, FixedEmbeddingComposite


def to_numpy_matrix(bqm, variable_order=None):
    """
    This method converts a BinaryQuadraticModel into a numpy matrix.

    Parameters
    ----------
    bqm: BinaryQuadraticModel
        Binary quadratic model class from D-Wave
    variable_order: Sequence
        Order of the variables.
    """

    if variable_order is None:
        if bqm.variables ^ range(bqm.num_variables):
            raise ValueError("variable_order must be provided for BQMs "
                             "that are not index-labelled")
        variable_order = range(bqm.num_variables)
    elif bqm.variables ^ variable_order:
        raise ValueError("variable_order does not match the BQM variables")

    if bqm.vartype is dimod.Vartype.SPIN:
        bqm = bqm.change_vartype(dimod.Vartype.BINARY, inplace=False)
    elif bqm.vartype is not dimod.Vartype.BINARY:
        raise RuntimeError("unexpected vartype")

    ldata, (irow, icol, qdata), _ = bqm.to_numpy_vectors(variable_order)

    # make sure it's upper triangular
    idx = irow > icol
    if idx.any():
        irow[idx], icol[idx] = icol[idx], irow[idx]

    dense = np.zeros((bqm.num_variables, bqm.num_variables),
                     dtype=bqm.dtype)
    dense[irow, icol] = qdata

    # set the linear
    np.fill_diagonal(dense, ldata)

    return dense


def convert_qubo(qubo, convert_to='BQM', variable_order=None):
    """
    Method for converting QUBO to desired type.
    """

    if convert_to == 'BQM':
        if isinstance(qubo, np.ndarray):
            qubo = BinaryQuadraticModel(qubo, 'BINARY')
        elif isinstance(qubo, qubolite.qubo):
            qubo = BinaryQuadraticModel(qubo.m, 'BINARY')
        elif isinstance(qubo, dict):
            qubo = BinaryQuadraticModel(qubo, 'BINARY')
        elif not isinstance(qubo, BinaryQuadraticModel):
            raise Exception("QUBO has to be of class qubo, numpy matrix or BinaryQuadraticModel!")
    elif convert_to == "Matrix":
        if isinstance(qubo, qubolite.qubo):
            qubo = qubo.m
        elif isinstance(qubo, dict):
            qubo = to_numpy_matrix(BinaryQuadraticModel(qubo, 'BINARY'),
                                   variable_order=variable_order)
        elif isinstance(qubo, BinaryQuadraticModel):
            qubo = to_numpy_matrix(qubo, variable_order=variable_order)
        elif not isinstance(qubo, np.ndarray):
            raise Exception("QUBO has to be of class qubo, numpy matrix or BinaryQuadraticModel!")
    else:
        raise Exception("convert_to has to be either 'BQM' or 'Matrix'!")
    return qubo


class DWaveAnnealing:
    """
    Class for D-Wave annealing.
    """

    properties = None
    parameters = None

    def __init__(self, token=None, region="eu-central-1", embedding=None, **config):
        self.d_wave = DWaveSampler(token=token, region=region, **config)
        embedding = self._get_embedding(embedding)
        if embedding is None:
            self.d_wave_embedding = EmbeddingComposite(self.d_wave)
        else:
            self.d_wave_embedding = FixedEmbeddingComposite(self.d_wave, embedding=embedding)
        self.parameters = self.d_wave.parameters
        self.properties = self.d_wave.properties

    @staticmethod
    def _get_embedding(embedding):
        if isinstance(embedding, str):
            try:
                # Read dictionary pkl file
                with open(embedding, 'rb') as fp:
                    embedding = pickle.load(fp)
                    print('Loaded embedding!')
            except Exception as e:
                print(e)
                embedding = None
        return embedding

    def sample_qubo(self, qubo, sample_set_flag=False, chain_method=None, **sampling_kwargs):
        qubo = convert_qubo(qubo)
        qubo_matrix = convert_qubo(qubo, convert_to='Matrix')
        if chain_method == 'max':
            chain_strength = np.max([np.abs(np.min(qubo_matrix)),
                                     np.abs(np.max(qubo_matrix))]) * 1.25
            sample_set = self.d_wave_embedding.sample_qubo(qubo_matrix,
                                                           chain_strength=chain_strength,
                                                           **sampling_kwargs)
        else:
            sample_set = self.d_wave_embedding.sample_qubo(qubo_matrix, **sampling_kwargs)
        if sample_set_flag:
            result = sample_set
        else:
            first = sample_set.first
            result = dimod.as_samples(first.sample)[0][0], first.energy
        return result

    def fix_embedding(self, embedding):
        self.d_wave_embedding = FixedEmbeddingComposite(self.d_wave, embedding=embedding)

    def get_name(self):
        return "d_wave"


def d_wave(Q, dWaveAnnealing, sample_set_flag=False, fix_embedding=False, **parameters):
    assert isinstance(dWaveAnnealing, DWaveAnnealing)
    if fix_embedding:
        if not isinstance(dWaveAnnealing.d_wave_embedding, FixedEmbeddingComposite):
            print('Find embedding...')
            sample_set = dWaveAnnealing.sample_qubo(Q, sample_set_flag=True, return_embedding=True,
                                                    **parameters)
            print('Found embedding!')
            embedding = sample_set.info['embedding_context']['embedding']
            with open(f'embedding.pkl', 'wb') as f:
                pickle.dump(dict(embedding), f)
                print('Embedding saved! Needs to be renamed!')
            dWaveAnnealing.fix_embedding(embedding)
            if sample_set_flag:
                result = sample_set
            else:
                first = sample_set.first
                result = dimod.as_samples(first.sample)[0][0], first.energy
        else:
            result = dWaveAnnealing.sample_qubo(Q, sample_set_flag=sample_set_flag, **parameters)
    else:
        result = dWaveAnnealing.sample_qubo(Q, sample_set_flag=sample_set_flag, **parameters)
    return result


if __name__ == '__main__':
    dwave = DWaveAnnealing(token="DEV-4dac0699f916fac35ed43bb467657db2e6a94126",region=None)
    Q = qubolite.qubo.random(n=4)
    sample_set = d_wave(Q, dwave, sample_set_flag=True, num_reads=10)
    print(sample_set)
