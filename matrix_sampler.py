# coding=utf-8

__author__ = 'yangzhenxi'
import numpy

def isnot_sym_index(index_arg):
    return index_arg[0] != index_arg[1]

def downsample(matrix_arg, matrix_indices, n, lowerbound, upperbound):
    """
    :param matrix_arg:
    :param matrix_indices:
    :param n:
    :return:
    """
    downsampled_value = matrix_arg.flatten()[::n]
    downsampled_indices = matrix_indices.flatten()[::n]

    bounded_indices = numpy.logical_and(downsampled_value >= lowerbound, downsampled_value < upperbound)

    v_is_sym_index = numpy.vectorize(isnot_sym_index)
    filtered_indices = numpy.logical_and(bounded_indices, v_is_sym_index(downsampled_indices))
    return numpy.extract(filtered_indices, downsampled_value), numpy.extract(filtered_indices, downsampled_indices)