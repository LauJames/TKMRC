#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : preprocess_tk.py
@Time     : 18-8-28 下午7:15
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import pandas as pd
import json
import jieba

path = './data0828.xlsx'


def excel2json():
    """
    input: excel files: 答案ID	问题类型	问题	段落	答案. all of them are string type
    transform the raw data to json type according to the DuReader README.md "## Raw Data Format"
    :return:
    """
    # 答案ID	问题类型	问题	段落	答案
    dataframe = pd.read_excel(path,
                              sheet_name='Sheet1',
                              usecols=[2, 3, 4],
                              header=0,
                              dtype={'答案': str, '段落': str, '问题': str})
    paragraphs = []
    questions = []
    answers = []

    for key, data in dataframe.iterrows():  # 返回key, rows.values()
        paragraphs.append(data['段落'])
        questions.append(data['问题'])
        answers.append(data['答案'])

    data_list = []
    for para, question, answer in zip(paragraphs, questions, answers):
        paras_json = [{"paragraphs": [text for text in para.split('|')]}]
        answers_json = [text for text in answer.split('|')]
        item_json = {"question": question, "documents": paras_json, "answers": answers_json}
        data_list.append(item_json)
    print(data_list)
    data_json = json.dumps(data_list, ensure_ascii=False)
    with open('tk_json.json', 'w', encoding='utf-8') as tk_json:
        tk_json.write(data_json)
    print(data_json)


def cut2tokens(text):
    """
    use jieba.cut do word segmentation, and return one line
    :param text:
    :return: tokens
    """
    cuted = []
    for cut_line in jieba.cut(text):
        if cut_line not in ['', '\n', ' ', '\t']:
            cuted.append(cut_line)
            # print(cut_line)
    # print(cuted)
    return cuted


def excel2json_seg():
    """
    input: excel files: 答案ID	问题类型	问题	段落	答案. all of them are string type
    transform the raw data to json type according to the DuReader README.md "## Raw Data Format"
    :return:
    """

    # 答案ID	问题类型	问题	段落	答案
    dataframe = pd.read_excel(path,
                              sheet_name='Sheet1',
                              usecols=[2, 3, 4],
                              header=0,
                              dtype={'答案': str, '段落': str, '问题': str})
    paragraphs = []
    questions = []
    answers = []

    for key, data in dataframe.iterrows():  # 返回key, rows.values()
        paragraphs.append(data['段落'])
        questions.append(data['问题'])
        answers.append(data['答案'])

    data_list = []
    for para, question, answer in zip(paragraphs, questions, answers):
        paras_and_seg_json = [{"paragraphs": [text.replace('\n', '') for text in para.split('|')],
                               "segmented_paragraphs": [cut2tokens(text) for text in para.split('|')]}]
        print(paras_and_seg_json)
        answers_json = [text for text in answer.split('|')]
        answers_seg_json = [cut2tokens(text) for text in answer.split('|')]
        question_seg = [cut2tokens(token) for token in jieba.cut(question)]
        item_json = {"question": question,
                     "segmented_question": question_seg,
                     "documents": paras_and_seg_json,
                     "answers": answers_json,
                     "segmented_answers": answers_seg_json}
        data_list.append(item_json)
    # print(data_list)
    data_json = json.dumps(data_list, ensure_ascii=False)
    with open('tk_json_cut.json', 'w', encoding='utf-8') as tk_json:
        tk_json.write(data_json)
    print(data_json)


def main():
    excel2json_seg()
    # result = cut2tokens("如您申请解除本合同，请填写解除合同申请书并向我们提供您的有效身份证件。\
    #             自我们收到解除合同申请书时起，本合同终止。我们自收到解除合同申请书之日起三十日内向您退还本合同的未满期净保险费（见8.29）。\
    #             您申请解除合同会遭受一定经济损失。")
    # print(result)


if __name__ == '__main__':
    main()
