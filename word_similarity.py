# coding=utf-8

__author__ = 'yangzhenxi'
from Levenshtein import distance, jaro
import numpy
import json
from jieba import posseg as pseg
from scipy.spatial import distance

def cut_HFM_data(arg_HFM_name_array):
    result_array = numpy.array([set()]*arg_HFM_name_array.size)
    for i in range(arg_HFM_name_array.size):
        #print arg_HFM_name_array[i]
        segs = pseg.cut(arg_HFM_name_array[i])
        result_array[i] = set()
        for w in segs:
            result_array[i].add(w.word)
    return result_array

def word_similarity(arg1, arg2):
    """
    :param arg1: input word 1 of unicode type
    :param arg2: input word 2 of unicode type
    :return: jaro similarity value of arg1 and arg2
    """
    return jaro(arg1, arg2)

def company_name_similarity_ht(arg1, arg2, tdf_weight_dict):
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
    vector1 = list()
    vector2 = list()
    for w in union_word_bag:
        is_penalty_word = tdf_weight_dict[w]['penalty'] == 'True'
        is_penalty_condition = not ((w in word_bag1) and (w in word_bag2))
        weight = float(tdf_weight_dict[w]['value'])
        if is_penalty_word & is_penalty_condition :
            weight *= float(tdf_weight_dict[w]['effect_value'])
        vector1.append((w in word_bag1) * float(1) * weight / float(len(word_bag1)))
        vector2.append((w in word_bag2) * float(1) * weight / float(len(word_bag2)))

    cosine_distance = distance.cosine(vector1, vector2)
    return 1 - round(cosine_distance, 10)

def set_similarity(arg1, arg2, tdf_weight=None):
    """
    :param arg1: numpy array of unicode string
    :param arg2: numpy array of unicode string
    :return: matrix of the similarity of the elements
    """
    similarity_matrix = numpy.zeros((arg1.size, arg2.size))
    similarity_matrix_indices = numpy.zeros((arg1.size, arg2.size), dtype=tuple)
    cut_arg1 = cut_HFM_data(arg1)
    cut_arg2 = cut_HFM_data(arg2)
    for i in range(arg1.size):
        print "iteration " + str(i)
        for j in range(arg2.size):
            similarity_matrix[i][j] = company_name_similarity_ht(cut_arg1[i], cut_arg2[j], tdf_weight)
            similarity_matrix_indices[i][j] = (i, j)
    return similarity_matrix, similarity_matrix_indices

if __name__ == '__main__':
    tdf_weitht = json.loads(open('tdf_dict', 'r+').read())
    word1 = ur'华润置地(成都)有限公司'
    word2 = ur'华润置地(成都)有限公司(PRC虚拟节点)'
    seg1 = pseg.cut(word1)
    cut_data1 = set()
    for w in seg1:
        cut_data1.add(w.word)
    seg2 = pseg.cut(word2)
    cut_data2 = set()
    for w in seg2:
        cut_data2.add(w.word)
    result = company_name_similarity_ht(cut_data1, cut_data2, tdf_weitht)


