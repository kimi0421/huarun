# coding=utf-8

__author__ = 'yangzhenxi'
from Levenshtein import distance, jaro
import numpy
import jieba.posseg as pseg
import jieba.analyse as anal


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

    seg_list1 = pseg.cut(arg1)
    for w in seg_list1:
        print w.word, w.flag

    seg_list2 = pseg.cut(arg2)
    for w in seg_list2:
        print w.word, w.flag
    return 1

def set_similarity(arg1, arg2):
    """
    :param arg1: numpy array of unicode string
    :param arg2: numpy array of unicode string
    :return: matrix of the similarity of the elements
    """
    similarity_matrix = numpy.zeros((arg1.size, arg2.size))
    similarity_matrix_indices = numpy.zeros((arg1.size, arg2.size), dtype=tuple)
    for i in range(arg1.size):
        for j in range(arg2.size):
            similarity_matrix[i][j] = word_similarity(arg1[i], arg2[j])
            similarity_matrix_indices[i][j] = (i, j)
    return similarity_matrix, similarity_matrix_indices

if __name__ == '__main__':
    result = company_name_similarity_ht(ur'辽宁（华润万家）生活超市有限公司', ur'陕西华润万家生活超市有限公司')
    print result


