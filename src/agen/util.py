'''
utilities for agen

Stanley Bak
Feb 2021
'''

from itertools import chain

import re
import onnx
import onnxruntime as ort

import numpy as np

def predict_with_onnxruntime(model_def, *inputs):
    'run an onnx model'
    
    sess = ort.InferenceSession(model_def.SerializeToString())
    names = [i.name for i in sess.get_inputs()]

    inp = dict(zip(names, inputs))
    res = sess.run(None, inp)

    #names = [o.name for o in sess.get_outputs()]

    return res[0]

def remove_unused_initializers(model):
    'return a modified model'

    new_init = []

    for init in model.graph.initializer:
        found = False
        
        for node in model.graph.node:
            for i in node.input:
                if init.name == i:
                    found = True
                    break

            if found:
                break

        if found:
            new_init.append(init)
        else:
            print(f"removing unused initializer {init.name}")

    graph = onnx.helper.make_graph(model.graph.node, model.graph.name, model.graph.input,
                        model.graph.output, new_init)

    onnx_model = make_model_with_graph(model, graph)

    return onnx_model

def get_io_nodes(onnx_model):
    'returns single input and output nodes'

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = [i.name for i in sess.get_inputs()]
    assert len(inputs) == 1, f"expected single onnx network input, got: {inputs}"
    input_name = inputs[0]

    outputs = [o.name for o in sess.get_outputs()]
    assert len(outputs) == 1, f"expected single onnx network output, got: {outputs}"
    output_name = outputs[0]

    g = onnx_model.graph
    inp = [n for n in g.input if n.name == input_name][0]
    out = [n for n in g.output if n.name == output_name][0]

    return inp, out

def make_model_with_graph(model, graph, ir_version=None, check_model=True):
    'copy a model with a new graph'

    onnx_model = onnx.helper.make_model(graph)
    onnx_model.ir_version = ir_version if ir_version is not None else model.ir_version
    onnx_model.producer_name = model.producer_name
    onnx_model.producer_version = model.producer_version
    onnx_model.domain = model.domain
    onnx_model.model_version = model.model_version
    onnx_model.doc_string = model.doc_string

    #print(f"making model with ir version: {model.ir_version}")
    
    if len(model.metadata_props) > 0:
        values = {p.key: p.value for p in model.metadata_props}
        onnx.helper.set_model_props(onnx_model, values)

    # fix opset import
    for oimp in model.opset_import:
        op_set = onnx_model.opset_import.add()
        op_set.domain = oimp.domain
        op_set.version = oimp.version

    if check_model:
        onnx.checker.check_model(onnx_model, full_check=True)

    return onnx_model

def glue_models(model1, model2):
    'glue the two onnx models into one'

    g1_in, g1_out = get_io_nodes(model1)
    g2_in, _ = get_io_nodes(model2)
    
    assert g1_out.name == g2_in.name, f"model1 output was {g1_out.name}, " + \
        f"but model2 input was {g2_in.name}"

    graph1 = model1.graph
    graph2 = model2.graph

    var_in = [g1_in]

    # sometimes initializers are added as inputs
    #for inp in graph2.input[1:]:
    #    var_in.append(inp)
    
    var_out = graph2.output

    combined_init = []
    names = []
    for init in chain(graph1.initializer, graph2.initializer):
        assert init.name not in names, f"repeated initializer name: {init.name}"
        names.append(init.name)

        combined_init.append(init)

    #print(f"initialier names: {names}")

    combined_nodes = []
    #names = []
    for n in chain(graph1.node, graph2.node):
        #assert n.name not in names, f"repeated node name: {n.name}"
        #names.append(n.name)

        combined_nodes.append(n)

    name = graph2.name
    graph = onnx.helper.make_graph(combined_nodes, name, var_in,
                       var_out, combined_init)

    #print(f"making model with inputs {inputs} / outputs {outputs} and nodes len: {len(keep_nodes)}")

    # use ir_version 4 because we don't add inputs as initializers
    onnx_model = make_model_with_graph(model2, graph, ir_version=4)

    return onnx_model

def read_vnnlib_simple(vnnlib_filename, num_inputs, num_outputs):
    '''process in a vnnlib file

    this is not a general parser, and assumes files are provided in a 'nice' format

    outputs:
    1. input ranges (box), list of pairs
    2. output matrix, mat * y <= rhs
    3. output rhs vector, mat * y <= rhs
    '''

    #; Unscaled Input 4: (960, 1200)
    #(assert (<= X_4 0.5))
    #(assert (>= X_4 0.3))

    #; output constraints (property 3, sat if CoC is minimal)
    #(assert (<= Y_0 Y_1))
    #(assert (<= Y_0 Y_2))
    #(assert (<= Y_0 Y_3))
    #(assert (<= Y_0 Y_4))

    input_box = []
    mat = []
    rhs_list = []

    r = re.compile(r"^\(assert \((<=|>=) (\S+) (\S+)\)\)$") 

    with open(vnnlib_filename, 'r') as f:
        lines = f.readlines()

    lines = [line.rstrip() for line in lines]

    assert len(lines) > 0

    for line in lines:
        groups = r.findall(line)

        if len(groups) == 0:
            continue

        groups = groups[0]
        assert len(groups) == 3
        
        op, first, second = groups

        if first.startswith("X_"):
            # Input constraints
            index = int(first[2:])

            if index == len(input_box):
                if len(input_box) > 0:
                    assert input_box[-1][0] != -np.inf, f"lower bound for X_{index-1} not set"
                    assert input_box[-1][1] != np.inf, f"upper bound for X_{index-1} not set"

                input_box.append([-np.inf, np.inf])

            assert index == len(input_box) - 1

            if op == "<=":
                input_box[-1][1] = float(second)
            else:
                input_box[-1][0] = float(second)

        else:
            # output constraint

            if op == ">=":
                # swap order
                first, second = second, first

            row = [0.0] * num_outputs
            rhs = 0.0

            # assume op is <=
            if first.startswith("Y_") and second.startswith("Y_"):
                index1 = int(first[2:])
                index2 = int(second[2:])

                row[index1] = 1
                row[index2] = -1
            elif first.startswith("Y_"):
                index1 = int(first[2:])
                row[index1] = 1
                rhs = float(second)
            else:
                assert second.startswith("Y_")
                index2 = int(second[2:])
                row[index2] = -1
                rhs = -1 * float(second)

            mat.append(row)
            rhs_list.append(rhs)

    assert len(input_box) == num_inputs
            
    return input_box, np.array(mat, dtype=float), np.array(rhs_list, dtype=float) 
