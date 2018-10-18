#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : TKMRC 
@File     : web_server9002.py
@Time     : 18-10-15 下午9:32
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tornado.ioloop
import tornado.web
import json
import os

from tf_version.inference import prepare, infer

vocab, search, ir_config, args, rc_model = prepare()


class QAHandler(tornado.web.RequestHandler):

    def get(self):
        self.render('index.html')

    def post(self):
        self.use_write()

    def use_write(self):
        query = self.get_argument('query')
        try:
            title, reference, answer = infer(query, vocab, search, ir_config, args, rc_model)
            json_data = {'question': str(query),
                         'ref_title': str(title),
                         'reference': str(reference),
                         'answer': str(answer)}
            print(json_data)
            self.write(json.dumps(json_data, ensure_ascii=False))  # ensure_ascii=False 保证返回的数据不以unicode形式显示
        except Exception as e:
            print(e)
            json_data = {'question': str(query),
                         'ref_title': "None",
                         'reference': "None",
                         'answer': '不好意思，这个问题超出了我的认知范围...'}
            self.write(json.dumps(json_data, ensure_ascii=False))


class AnswerHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('index.html')

    def post(self, *args, **kwargs):
        self.use_render()

    def use_render(self):
        query = self.get_argument('query')  # get/post混合方式获取传递来的数据
        try:
            title, reference, answer = infer(query, vocab, search, ir_config, args, rc_model)
            json_data = {'question': str(query),
                         'ref_title': str(title),
                         'reference': str(reference),
                         'answer': str(answer)}
            print(json_data)
            self.render('answer.html',
                        question=query,
                        ref_title=json_data['ref_title'],
                        reference=json_data['reference'],
                        answer=json_data['answer'])
        except Exception as e:
            print(e)
            json_data = '不好意思，这个问题超出了我的认知范围...'
            self.write(json_data)


class DemoHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('online.html')


def make_app():
    setting = dict(
        template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
    )
    print(setting)
    return tornado.web.Application([(r'/MRCQA', QAHandler),
                                    # (r'/Answer', AnswerHandler),
                                    (r'/Answer', DemoHandler)
                                    ],
                                   **setting
                                   )


if __name__ == '__main__':
    app = make_app()
    app.listen(9002)
    tornado.ioloop.IOLoop.current().start()
