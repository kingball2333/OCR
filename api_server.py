import time
from datetime import datetime
from urllib.parse import quote
import re

from flask import Flask, request, jsonify
from flask.views import MethodView
from intelab_python_sdk.logger import log_init, log
from inference import inference


class ScaleAnalyzer(MethodView):
    def post(self):
        try:
            # 获取请求参数
            args = request.form or request.get_json()
            pict_url = args.get('pictureUrl')
            log.info("图片地址: {}".format(pict_url))

            # 确保 URL 安全编码
            if pict_url:
                # 对 URL 进行编码，保留常见的 URL 特殊字符
                encoded_url = quote(pict_url, safe=':/?&=')
            else:
                raise ValueError("未提供图片地址")

        except Exception as e:
            log.exception("参数错误: {}".format(e))
            response = {'code': 400, 'result': None, 'msg': '参数错误'}
            return jsonify(response), 400

        if not pict_url:
            response = {'code': 401, 'result': None, 'msg': '流获取失败'}
            log.info("流获取失败")
            return jsonify(response), 401

        t0 = time.time()
        try:
            # 调用 inference 函数进行推理
            scale_result = inference(encoded_url)  # 使用编码后的 URL 进行推理

            # 校验处理数字小数点的过滤
            if scale_result:
                # 若结果中包含非数字和小数点的字符，提取数字和小数点部分
                if re.search(r'[^0-9.]', scale_result):
                    scale_result = ''.join(re.findall(r'[0-9.]', scale_result))

                # 如果结果中只有数字和小数点就返回结果
                if re.match(r'^[0-9.]+$', scale_result):
                    response = {'code': 200,
                                'result': {'scale_result': scale_result},
                                'msg': 'ok'}

                else:
                    # 如果没有数字或小数点就返回 null
                    response = {'code': 200, 'result': None, 'msg': '推理结果不合法'}

            else:
                # 如果没有结果，返回 null
                response = {'code': 200, 'result': None, 'msg': '推理结果不合法'}


        except Exception as e:
            log.exception("推理错误: {}".format(e))
            response = {'code': 200, 'result': None, 'msg': '推理错误'}

        log.info("接口返回值为：{}".format(response))
        log.info("接口耗时: {:.2f} 秒".format(time.time() - t0))
        return jsonify(response), response['code']


def create_app(log_path):
    app = Flask(__name__)

    # 注册类视图，并指定允许的 HTTP 方法
    app.add_url_rule('/api/ai/scale/analyzer', view_func=ScaleAnalyzer.as_view('scale_analyzer'), methods=['POST'])

    @app.before_request
    def log_request():
        log.info(' "{} {}"'.format(request.path, request.method))

    # 初始化日志
    log_init('api-server', debug=False, log_path=log_path, backupCount=7)
    return app


if __name__ == '__main__':
    # 使用原始字符串避免路径转义问题
    app = create_app(r'E:\electronic_scale\log')
    app.run(debug=False, host='0.0.0.0', port=5001)

