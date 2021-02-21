'''
random test generation to find unsat systems

Stanley Bak, Feb 2021
'''

import sys
import numpy as np

import onnx

from agen.util import read_vnnlib_simple, predict_with_onnxruntime, remove_unused_initializers, get_io_nodes

def run_tests(onnx_filename, vnnlib_filename, num_trials):
    '''execute the model and its conversion as a sanity check

    returns string to print to output file
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

    print(f"Testing onnx model with {num_inputs} inputs and {num_outputs} outputs")

    input_box, prop_mat, prop_rhs = read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs)

    # use random input to validate conversion

    inp, _ = get_io_nodes(onnx_model)
    inp_shape = tuple(d.dim_value if d.dim_value != 0 else 1 for d in inp.type.tensor_type.shape.dim)
    res = 'unknown'

    for trial in range(num_trials):

        input_list = []

        for lb, ub in input_box:
            r = np.random.random()

            input_list.append(lb + (ub - lb) * r)

        random_input = np.array(input_list, dtype=np.float32)
        random_input.shape = inp_shape # resshape order might matter for more than 1-d input

        output = predict_with_onnxruntime(onnx_model, random_input)

        flat_out = np.ravel(output)

        vec = prop_mat.dot(flat_out)

        sat = np.all(vec <= prop_rhs)

        if sat:
            print(f"Trial #{trial + 1} found sat case with input {input_list} and output {list(flat_out)}")
            res = 'sat'
            break

    print(f"Result: {res}")

    return res

def main():
    'main entry point'

    trials = 1000
    seed = 0

    assert len(sys.argv) >= 4, "expected at least 3 args: <onnx-filename> <vnnlib-filename> <output-filename> " + \
        f"[<trials>] [<seed>] got {len(sys.argv)}"

    onnx_filename = sys.argv[1]
    vnnlib_filename = sys.argv[2]
    output_filename = sys.argv[3]
    
    if len(sys.argv) > 4:
        trials = int(sys.argv[4])

    if len(sys.argv) > 5:
        seed = int(sys.argv[5])

    print(f"doing {trials} random trials with random seed {seed}")

    np.random.seed(seed)

    res = run_tests(onnx_filename, vnnlib_filename, trials)

    with open(output_filename, 'w') as f:
        f.write(res)

if __name__ == "__main__":
    main()
