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
import tqdm
import json

path = './data0828.xlsx'


def main():
    datas = pd.read_excel(path)
    paragrphs = []
    questions = []
    answers = []
    for data in datas:
        paragrphs.append(data["段落"])
        questions.append(data["问题"])
        answers.append(data["答案"])
    data_list = []
    for para, question, answer in paragrphs, questions, answers:
        paras_json = [{"paragraphs": [text for text in para]}]
        answers_json = [text for text in answer]
        item_json = {"question": question, "documents": paras_json, "answers": answers_json}
        data_list.append(item_json)
    print(data_list)
    data_json = json.dumps(data_list)
    print(data_json)


if __name__ == '__main__':
    main()
