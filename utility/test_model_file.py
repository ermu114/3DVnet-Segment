import tensorflow as tf

def func_01():
    path = 'D:\\tmp\\01\\func_01_model'
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.save(sess, path)
    pass

def func_02():
    path = 'D:\\tmp\\01\\func_01_model'
    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path)
        print(sess.run(result))
    pass

def func_03():
    path = 'D:\\tmp\\01\\func_01_model'

    saver = tf.train.import_meta_graph(path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, path)
        graph = tf.get_default_graph()
        print(sess.run(graph.get_tensor_by_name('v1:0')))
        print(sess.run(graph.get_tensor_by_name('v2:0')))
        print(sess.run(graph.get_tensor_by_name('add:0')))
    pass

from tensorflow.python.framework import graph_util
def func_04():
    path = 'D:\\tmp\\01\\func_01_model'

    v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
    v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
    result = v1 + v2
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
        with tf.gfile.GFile(path + '.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
    pass

def func_05():
    path = 'D:\\tmp\\01\\func_01_model'

    with tf.Session() as sess:
        with tf.gfile.GFile(path + '.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            result = tf.import_graph_def(graph_def, return_elements=['add:0'])
            print(sess.run(result))
    pass

def func_06():
    path = 'D:\\tmp\\01\\func_01_model'

    output_node_names = "v1/v2/add"
    saver = tf.train.import_meta_graph(path + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, path)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split('/')
        )

        with tf.gfile.GFile(path + '.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        for op in graph.get_operations():
            print(op.name, op.values())
    pass

if __name__ == '__main__':
    # func_01()
    # func_02()
    # func_03()
    # func_04()
    # func_05()
    # func_06()
    pass