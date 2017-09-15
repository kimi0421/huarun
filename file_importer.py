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
import json

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

def down_sample_array(arg_name_array, N):
    result_array = arg_name_array[0::N]
    result_indices = range(0, arg_name_array.size, N)
    return result_array, result_indices

def preprocess_SIS_data(arg_SIS_df):
    # slice by label 'CHINESENAME'
    result_df = arg_SIS_df.loc[:, 'CHINESENAME']
    return result_df

def print_samples(sample_value,
                  sample_indices,
                  prefix_1,
                  name_array_1,
                  index_array_1,
                  prefix_2,
                  name_array_2,
                  index_array_2,
                  file_name):
    f = open(file_name, 'w')
    for i in range(sample_value.size):
        name_1 = name_array_1[index_array_1[sample_indices[i][0]]]
        index_1 = index_array_1[sample_indices[i][0]]
        name_2 = name_array_2[index_array_2[sample_indices[i][1]]]
        index_2 = index_array_2[sample_indices[i][1]]
        similarity_value = sample_value[i]
        f.write('---------------------------------------------\n')
        str_to_write = unicode(prefix_1) + ur': index_1: ' + unicode(index_1) + ur'name_1: ' + name_1 + '\n'
        f.write(str_to_write.encode('utf8'))
        str_to_write = unicode(prefix_2) + ur': index_2: ' + unicode(index_2) + ur'name_2 : ' + name_2 + '\n'
        f.write(str_to_write.encode('utf8'))
        str_to_write = r'similarity value: ' + str(similarity_value) + '\n'
        f.write(str_to_write)
    f.close()

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
    HFM_downsampled_data, HFM_downsampled_indices = down_sample_array(valid_HFM_result_with_simplied_name, 5)
    SIS_downsampled_data, SIS_downsampled_indices = down_sample_array(valid_SIS_result_with_simplied_name, 5)
    print "shape of HFM_sampled_data " + str(HFM_downsampled_data.size)
    print "shape of SIS_sampled_data " + str(SIS_downsampled_data.size)

    tdf_weight = json.loads(open('tdf_dict2', 'r+').read())
    similartity_matrix, matindices = set_similarity(arg1=HFM_downsampled_data,
                                                    arg2=SIS_downsampled_data,
                                                    tdf_weight=tdf_weight)

    # evaluate our model
    evaluate_result(similartity_matrix)

    # downsample some match records for insight
    for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        downsampled_value, downsampled_indices = downsample(similartity_matrix, matindices, 1, i, i+0.1)
        print "number in bucket "+str(i) + '_'+str(i+0.1)+' is: '+str(downsampled_value.size)
        print_samples(downsampled_value,
                      downsampled_indices,
                      "HFM",
                      valid_HFM_result_with_simplied_name,
                      HFM_downsampled_indices,
                      "SIS",
                      valid_SIS_result_with_simplied_name,
                      SIS_downsampled_indices,
                      'samples_' + str(i) + '_'+str(i+0.1)+'_HFM_SIS_cosine_distance4.txt')

    # print the distribution of the value in similarity_matrix
    pyplot.hist(similartity_matrix.flatten(), 50)
    pyplot.show()




