# coding=utf-8

__author__ = 'yangzhenxi'
import pandas
import re
import numpy
from matplotlib import pyplot
from word_similarity import set_similarity
from matrix_sampler import downsample
from result_evaluator import evaluate_result
import jieba.posseg as pseg

def is_simplified_chinese(arg):
    """check if arg containing only simplified chinese
    """
    pattern = ur'^[\u4e00-\u9fff]+$'
    return re.match(pattern, arg) != None

def import_data_from_csv(addr, encoding, index_col):
    f = open(addr)
    df = pandas.read_csv(filepath_or_buffer=f, index_col = index_col, encoding=encoding)
    f.close()
    return df

def import_data_from_excel(addr, index_col, sheetname=0):
    f = open(addr)
    df = pandas.read_excel(io=addr, sheetname=sheetname, index_col=index_col)
    f.close()
    return df

def preprocess_HFM_data(arg_HFM_df):
    # remove record with 'share' == TRUE and slice by label '公司名-简体'
    result_df = arg_HFM_df.loc[arg_HFM_df[ur'share'].isnull(), [ur'公司名-简体']]
    return result_df

def cut_HFM_data(arg_HFM_name_array):
    result_array = numpy.array([set()]*arg_HFM_name_array.size)
    for i in range(arg_HFM_name_array.size):
        segs = pseg.cut(arg_HFM_name_array[i])
        result_array[i] = set()
        for w in segs:
            result_array[i].add(w.word)
    return result_array

def down_sample_HFM_array(arg_HFM_name_array, N):
    result_array = arg_HFM_name_array[0::N]
    result_indices = range(0, arg_HFM_name_array.size, N)
    return result_array, result_indices

def preprocess_SIS_data(arg_SIS_df):
    # slice by label 'CHINESENAME'
    result_df = arg_SIS_df.loc[:, ['CHINESENAME']]
    return result_df

if __name__ == '__main__':
    data_set1 = import_data_from_csv(addr='huarun/HFM_data0703.csv', encoding='utf-8', index_col=False)
    data_set2 = import_data_from_csv(addr='huarun/SIS_Entity.csv', encoding='gbk', index_col=False)
    data_set3 = import_data_from_excel(addr='huarun/FMS数据_SBU_source.xls', index_col=False, sheetname=1)
    data_set4 = import_data_from_excel(addr='huarun/FMS数据_SBU_source.xls', index_col=False, sheetname=2)

    HFM_preprocessed_data = preprocess_HFM_data(data_set1)
    valid_HFM_result_with_simplied_name = HFM_preprocessed_data[HFM_preprocessed_data.notnull()].as_matrix().flatten()

    SIS_preprocessed_data = preprocess_SIS_data(data_set2)
    valid_SIS_result_with_simplied_name = SIS_preprocessed_data[SIS_preprocessed_data.notnull()].as_matrix().flatten()

    FMS1_result_with_simplied_name = data_set3[ur'机构描述']
    valid_FMS1_result_with_simplied_name = FMS1_result_with_simplied_name[FMS1_result_with_simplied_name.notnull()].as_matrix()
    FMS2_result_with_simplied_name = data_set4[ur'机构描述']
    valid_FMS2_result_with_simplied_name = FMS2_result_with_simplied_name[FMS2_result_with_simplied_name.notnull()].as_matrix()

    # very first model by return the similarity matrix of two sets of names
    HFM_cut_data, HFM_cut_indices = down_sample_HFM_array(cut_HFM_data(valid_HFM_result_with_simplied_name), 10)
    print "shape of HFM_cut_data " + str(HFM_cut_data.size)
    similartity_matrix, matindices = set_similarity(HFM_cut_data, HFM_cut_data)

    # evaluate our model
    evaluate_result(similartity_matrix)

    # downsample some match records for insight
    downsampled_value, downsampled_indices = downsample(similartity_matrix, matindices, 1, 0.9, 1)
    # print samples to file
    f = open(r'samples_0.9_1_HFM_HFM_cosine_distance.txt', 'w')
    for i in range(downsampled_value.size):
        HFM_name_1 = valid_HFM_result_with_simplied_name[HFM_cut_indices[downsampled_indices[i][0]]]
        HFM_index_1 = downsampled_indices[i][0]
        HFM_name_2 = valid_HFM_result_with_simplied_name[HFM_cut_indices[downsampled_indices[i][1]]]
        HFM_index_2 = downsampled_indices[i][1]
        similarity_value = downsampled_value[i]
        f.write('---------------------------------------------\n')
        str_to_write = ur'HFM_index_1: ' + unicode(HFM_index_1) + ur'HFM_name_1: ' + HFM_name_1 + '\n'
        f.write(str_to_write.encode('utf8'))
        str_to_write = ur'HFM_index_2: ' + unicode(HFM_index_2) + ur'HFM_name_2 : ' + HFM_name_2 + '\n'
        f.write(str_to_write.encode('utf8'))
        str_to_write = r'similarity value: ' + str(similarity_value) + '\n'
        f.write(str_to_write)
    f.close()

    # print the distribution of the value in similarity_matrix
    pyplot.hist(similartity_matrix.flatten(), 50)
    pyplot.show()




