############################
## TensorFlow版本: '1.14.0'
## Keras版本: '2.2.5'
## 测试通过
## TensorFlow2.0以上有问题
############################
import os
import os.path as osp

from keras.models import load_model
import tensorflow as tf
from keras import backend as K


#转换函数
def h5_to_pb(h5_model, output_dir, model_name, out_prefix='output_', log_tensorboard=False):
    if osp.exists(output_dir) == False:
        os.mkdir(output_dir)
    out_nodes = []
    print('h5_model output is: ', h5_model.outputs)
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.output[i], out_prefix + str(i + 1))

    print('out_nodes is: ')
    print(out_nodes)
    print('h5 output is: ')
    print(h5_model.outputs)
    sess = K.get_session()
    from tensorflow.python.framework import graph_util, graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name = model_name, as_text = False)
    if log_tensorboard:
        from tensorflow.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


if __name__ == '__main__':

    # 'PNET.h5', 'RNET.h5', 'ONET.h5, FACENET.h5'
    
    #路径参数
    input_path = './model_data'
    weight_file = 'FACENET.h5'
    weight_file_path = osp.join(input_path, weight_file)
    output_graph_name = weight_file[:-3] + '.pb'
    print(output_graph_name)
    
    #输出路径
    output_dir = osp.join(os.getcwd(), "./model_data")
    #加载模型
    h5_model = load_model(weight_file_path)
    h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
    print('model {} saved'.format(weight_file[:-3]))

