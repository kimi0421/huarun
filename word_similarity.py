# coding=utf-8

__author__ = 'yangzhenxi'
from Levenshtein import distance, jaro
import numpy
import jieba.posseg as pseg
from scipy.spatial import distance

def word_similarity(arg1, arg2):
    """
    :param arg1: input word 1 of unicode type
    :param arg2: input word 2 of unicode type
    :return: jaro similarity value of arg1 and arg2
    """
    arg1 = arg1.replace(u'有限公司', u'')
    arg2 = arg2.replace(u'有限公司', u'')
    return jaro(arg1, arg2)

def company_name_similarity_ht(arg1, arg2):
    """
    use a hand tuned model to check the similarity of company names
    if the most important segments are the same or similar, then
    the companies may have the higher similiarity value. use TF-IDF algorithm
    to tune down the significance of some frequently appeared segments
    :param arg1:
    :param arg2:
    :return:
    """

    word_bag1 = arg1
    word_bag2 = arg2
    union_word_bag = word_bag1 | word_bag2

    # vector construction
    size_of_union = len(union_word_bag)
    vector1 = numpy.zeros(size_of_union)
    vector2 = numpy.zeros(size_of_union)
    vector1 = [(w in word_bag1) * float(1) for w in union_word_bag]
    vector2 = [(w in word_bag2) * float(1) for w in union_word_bag]

    cosine_distance = distance.cosine(vector1, vector2)
    return 1 - round(cosine_distance, 10)

def set_similarity(arg1, arg2):
    """
    :param arg1: numpy array of unicode string
    :param arg2: numpy array of unicode string
    :return: matrix of the similarity of the elements
    """
    similarity_matrix = numpy.zeros((arg1.size, arg2.size))
    similarity_matrix_indices = numpy.zeros((arg1.size, arg2.size), dtype=tuple)
    for i in range(arg1.size):
        print "iteration " + str(i)
        for j in range(arg2.size):
            similarity_matrix[i][j] = company_name_similarity_ht(arg1[i], arg2[j])
            similarity_matrix_indices[i][j] = (i, j)
    return similarity_matrix, similarity_matrix_indices

if __name__ == '__main__':
    result = company_name_similarity_ht(ur'通化华润燃气(香港)有限公司', ur'海城华润燃气(香港)有限公司')
    print result


