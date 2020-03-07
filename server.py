# import logging
# import flask
# from flask import jsonify
# from flask import request

# import infer

# app = flask.Flask(__name__)

# @app.route('/', methods=['GET'])
# def healthCheck():
#     logging.info("Health check ping received")
#     return jsonify({'status': 'healthy'}), 200

# @app.route('/api/2d', methods=['POST'])
# def parseIntent():
#     imagefile = request.files['image']

#     inference.infer(imgfile=args.input_image,
#                     conf_threshold=args.conf_threshold)

#     data = flask.request.form  # is a dictionary
#     sentence = data['sentence']
#     word = data['word']
#     answer = bert.predict(sentence, word)

#     logging.info("word: " + answer[0])
#     return jsonify({'def': answer[0]}), 200

# if __name__ == '__main__':
#     logging.info("Starting server...")
#     inference = InferenceEngine(
#         model_json=MODEL_DIR + "net_arch.json", model_weights=MODEL_DIR + "weights_epoch96.h5")

#     app.run(host="0.0.0.0", port=5000)
