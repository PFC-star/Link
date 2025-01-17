

import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto
import onnxruntime as rt

def network_construct():

    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 3, 3])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 2, 2])

    # Make MaxPool Node
    node_def = onnx.helper.make_node(
        'MaxPool',
        inputs=['X'],
        outputs=['Y'],
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[1, 1, 1, 1]   # Top、Left、Bottom、Right
    )

    # Make Graph
    graph_def = helper.make_graph(
        [node_def],
        'test-MaxPool',
        [X],
        [Y]
    )

    # Make model
    model_def = helper.make_model(
        graph_def,
        producer_name='yang'
    )

    # Check & Save Model
    onnx.checker.check_model(model_def)
    onnx.save(model_def, 'MaxPool.onnx')


def  model_infer():

    # Infer Model
    model = onnx.load('MaxPool.onnx')
    print(model.ir_version)
    # 修改 IR 版本为 9
    model.ir_version = 9

    # 保存新的模型
    onnx.save(model, 'MaxPool_v9.onnx')

    # 确认 IR 版本已更新
    print("Updated IR version:", model.ir_version)
    sess = rt.InferenceSession('MaxPool_v9.onnx')

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_data = [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]
    input_data = np.array(input_data, dtype=np.float32)

    result = sess.run([output_name], {input_name: input_data})
    print(result)

def main():
    network_construct()
    model_infer()

if __name__ == '__main__':
    main()
