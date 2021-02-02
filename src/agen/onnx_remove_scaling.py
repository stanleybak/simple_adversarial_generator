'''
scales an onnx network with respect to a vnnlib file

this creates a new onnx network so that the input range (-1, 1) for each input corresponds to the full input range
given in the vnnlib file

Stanley Bak, Feb 2020
'''

import sys

import numpy as np

import onnx
from onnx import TensorProto
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxAdd

from util import read_vnnlib_simple, predict_with_onnxruntime, remove_unused_initializers, get_io_nodes, \
                 glue_models, make_model_with_graph

def model_convert(onnx_filename, vnnlib_filename, output_filename):
    '''make the model

    returns input_box
    '''

    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model, full_check=True)
    onnx_model = remove_unused_initializers(onnx_model)

    inp, out = get_io_nodes(onnx_model)
    
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    out_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in out.type.tensor_type.shape.dim)

    num_inputs = 1
    num_outputs = 1

    for n in inp_shape:
        num_inputs *= n

    for n in out_shape:
        num_outputs *= n

    print(f"Converting onnx model with num inputs: {num_inputs}, num_outputs: {num_outputs}")

    input_box, _, _ = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)
    
    mean_list = [(a+b)/2 for a, b in input_box]
    slope_list = [(b-a)/2 for (a, b) in input_box]

    b_mat = np.diag(slope_list).astype(np.float32)
    c_mat = np.array(mean_list, dtype=np.float32)

    b_mats = np.array([b_mat])
    c_mats = np.array([c_mat])

    #while len(b_mats.shape) != len(inp_shape):
    #    b_mats = np.expand_dims(b_mats, axis=0)

    while len(c_mats.shape) != len(inp_shape):
        c_mats = np.expand_dims(c_mats, axis=0)

    #assert b_mats.shape == inp_shape, f"b_mats shape: {b_mats.shape}, inp_shape: {inp_shape}"
    assert c_mats.shape == inp_shape, f"c_mats shape: {c_mats.shape}, inp_shape: {inp_shape}"

    old_input_name = inp.name
    input_name = 'new_input'

    matmul_node = OnnxMatMul(input_name, b_mats)
    add_node = OnnxAdd(matmul_node, c_mats, output_names=[old_input_name])

    i = onnx.helper.make_tensor_value_info('i', TensorProto.FLOAT, inp_shape)

    # test matmul model
    # zero should map to the mean
    # one should map to the max
    
    prefix_model = add_node.to_onnx({input_name: i})
    onnx.checker.check_model(prefix_model)
    
    zero_in = np.zeros(inp_shape, dtype=np.float32)
    o = predict_with_onnxruntime(prefix_model, zero_in)

    assert np.allclose(o, mean_list) # zeros should map to mean

    one_in = np.ones(inp_shape, dtype=np.float32)
    o = predict_with_onnxruntime(prefix_model, one_in)

    max_list = [dim[1] for dim in input_box]
    assert np.allclose(o, max_list) # ones should map to max

    # only shapes are used
    model2_def = add_node.to_onnx({input_name: i})
    onnx.checker.check_model(model2_def)

    combined = glue_models(model2_def, onnx_model)

    onnx.save_model(combined, output_filename)
    print(f"Saved converted model to: {output_filename}")

    return input_box

def model_execute(onnx_filename, output_filename, input_box):
    'execute the model and its conversion as a sanity check'

    # use random input to validate conversion

    onnx_model = onnx.load(onnx_filename)
    onnx_model = remove_unused_initializers(onnx_model)

    inp, _ = get_io_nodes(onnx_model)
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)

    np.random.seed(0) # deterministic random

    input_list = []
    scaled_input_list = []

    for lb, ub in input_box:
        #r = np.random.random()
        r = 0.0

        input_list.append(lb + (ub - lb) * r)
        scaled_input_list.append(2*r - 1.0)

    random_input = np.array(input_list, dtype=np.float32)
    random_input.shape = inp_shape # resshape order might matter for more than 1-d input

    output1 = predict_with_onnxruntime(onnx_model, random_input)

    random_scaled_input = np.array(scaled_input_list, dtype=np.float32)
    random_scaled_input.shape = inp_shape # resshape order might matter for more than 1-d input
    out_model = onnx.load(output_filename)
    output2 = predict_with_onnxruntime(out_model, random_scaled_input)

    assert np.allclose(output1, output2), "execution differs: {output1} and {output2}"

    print("Random execution matches.")

def main():
    'main entry point'

    assert len(sys.argv) == 4, "expected 3 args: <onnx-filename> <vnnlib-filename> <output-filename>, " + \
        f"got {len(sys.argv)}"
    
    onnx_filename = sys.argv[1]
    vnnlib_filename = sys.argv[2]
    output_filename = sys.argv[3]

    assert onnx_filename.endswith(".onnx")
    assert vnnlib_filename.endswith(".vnnlib")
    assert output_filename.endswith(".onnx")

    input_box = model_convert(onnx_filename, vnnlib_filename, output_filename)
    
    model_execute(onnx_filename, output_filename, input_box)

if __name__ == '__main__':
    main()
    
