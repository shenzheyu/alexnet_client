import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import time
import cv2
import numpy as np

tf.app.flags.DEFINE_string('server', '127.0.0.1:8500', 'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def batch_inference(server):
    for batch_size_index in range(0, 10):
        batch_size = 2 ** batch_size_index
        print("Batch size = %d: (sec per image)" % batch_size)
        mean = 0
        first = 0
        second = 0
        for test_index in range(30):
            time = do_inference(server, batch_size) / batch_size
            # print("    test %d: %f" % (test_index + 1, time))
            if test_index == 0:
                first = time
            elif test_index == 1:
                second = time
                mean += time
            else:
                mean += time
        print("    mean: %f" % (mean / 29))
        print("    first variance: %f" % ((first - second) * batch_size))


def do_inference(server, batch_size):
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'alexnet'
    request.model_spec.signature_name = 'inputs'

    # Going to read the image
    image = cv2.imread('./examples/dog.png')
    image = image.astype('f')

    input = np.expand_dims(image, axis=0)
    if batch_size == 1:
        inputs = input
    else:
        inputs = np.append(input, input, axis=0)
        for _ in range(batch_size - 2):
            inputs = np.append(inputs, input, axis=0)

    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto
                                     (inputs, shape=inputs.shape))
    start = time.time()
    stub.Predict(request, 10.25)  #  seconds
    end = time.time()
    return end - start


def main(_):
    if not FLAGS.server:
        print('please specify server -server host:port')
        return
    batch_inference(FLAGS.server)


if __name__ == '__main__':
    tf.app.run()