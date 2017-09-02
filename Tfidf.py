#!/usr/bin/python
# coding=utf-8
# --------------------------------------------
# __author__ = "Kimi Jin"
# __copyright__ = "Copyright 2016, PrettyYes"
# __version__ = "1.0.1"
# __maintainer__ = "Kimi Jin"
# __email__ = "kimi@prettyyes.com"
# __created__ = "2/25/16"
# __status__ = "Production"
# --------------------------------------------
import file_importer
import jieba
import numpy as np
import operator
from collections import Counter
from gensim.corpora import Dictionary
import json

FILE1 = u'huarun/HFM_data0703.csv'
FILE2 = u'huarun/FMS数据_SBU_source'
FILE3 = u'huarun/SIS_Entity.csv'


class Tfidf:

    def __init__(self, words_bag):
        self.words_bag = words_bag
        return

    def _process_data(self):
        words = []
        for company in self.words_bag:
            try:
                words.extend(self._get_segmented(company))
            except Exception as e:
                print(e)
        return words

    def get_word_count(self):
        words = self._process_data()
        words_dict = dict(Counter(words))
        return sorted(words_dict.items(), key=operator.itemgetter(1))

    def get_small_group(self):
        dictionary = Dictionary(self.words_bag)

    @staticmethod
    def _get_segmented(words):
        seg_list = list(jieba.cut(words, cut_all=False))
        return seg_list

if __name__ == '__main__':
    df1 = file_importer.import_data_from_csv(addr='huarun/HFM_data0703.csv', encoding='utf-8', index_col=False)
    df2 = file_importer.import_data_from_csv(addr='huarun/SIS_Entity.csv', encoding='gbk', index_col=False)
    df3 = file_importer.import_data_from_excel(addr='huarun/FMS数据_SBU_source.xls', index_col=False, sheetname=1)
    df4 = file_importer.import_data_from_excel(addr='huarun/FMS数据_SBU_source.xls', index_col=False, sheetname=2)
    company_names = []
    company_names.extend(df1[u'公司名-简体'].tolist())
    company_names.extend(df2['CHINESENAME'].tolist())
    company_names.extend(df3[u'机构描述'].tolist())
    company_names.extend(df4[u'机构描述'].tolist())
    tf = Tfidf(company_names)
    words_count = tf.get_word_count()
    total_count = len(company_names)
    tdf_dict = {}
    with open('word_count.txt', 'w') as w:
        for word_with_count in words_count:
            line = word_with_count[0] + '\t' + str(np.log(total_count/(word_with_count[1] * 1.0 + 1))) + '\n'
            tdf_dict[word_with_count[0]] = str(np.log(total_count/(word_with_count[1] * 1.0 + 1)))
            w.write(line.encode('utf-8'))
    w.close()
    json.dump(tdf_dict, open('tdf_dict', 'w'))
    import ipdb; ipdb.set_trace()

