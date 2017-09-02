# coding=utf-8

__author__ = 'yangzhenxi'
import numpy

def evaluate_result(similarity_matrix_arg, label_arg=None):
    """
    :param similarity_matrix_arg:
    :param label_arg:
    :return:
    """
    flattened_matrix = similarity_matrix_arg.flatten()
    print "shape of similarity matrix"
    print similarity_matrix_arg.shape
    exact_match = numpy.count_nonzero(flattened_matrix==1)
    print "number of exact match (similarity = 1): " + str(exact_match)
    confident_match = numpy.count_nonzero(numpy.logical_and(flattened_matrix < 1, flattened_matrix >= 0.9))
    print "number of confident match (0.9 <= similarity < 1): " + str(confident_match)
    confused_match = numpy.count_nonzero(numpy.logical_and(flattened_matrix < 0.9, flattened_matrix >= 0.8))
    print "number of confused match (0.8 <= similarity < 0.9): " + str(confused_match)
    not_match = numpy.count_nonzero(flattened_matrix < 0.8)
    print "number of not match (similarity < 0.8): " + str(not_match)
    return