import os
import onnx
import onnxsim

ONNX_MODEL_PATH = r'../model/yolox_nano.onnx'
ONNX_OPT_SAVE_PATH = os.path.splitext(ONNX_MODEL_PATH)[0] + '_simple.onnx'


def add_node_inputs(in_graph, node_idx, input_name=None, input_shape=None):
    input_info = onnx.helper.make_tensor_value_info(input_name, elem_type=onnx.TensorProto.FLOAT, shape=input_shape)
    in_graph.input.insert(0, input_info)
    in_graph.node[node_idx].input[0] = input_name
    return in_graph


def add_node_outputs(in_graph, node_idx, output_name=None, output_shape=None):
    output_info = onnx.helper.make_tensor_value_info(output_name, elem_type=onnx.TensorProto.FLOAT, shape=output_shape)
    in_graph.output.insert(0, output_info)
    in_graph.node[node_idx].output[0] = output_name
    return in_graph


def cut_input_nodes(in_graph, input_nodes_names=None, input_names=None):
    nodes = in_graph.node
    graph_input_num = len(in_graph.input)
    # 重新设置graph input的names
    input_names_iter = iter(input_names)
    # 将node name与idx对应
    nodes_dict = {nodes[i].name: i for i in range(len(nodes))}
    # 将node output与idx对应
    nodes_outputs_dict = {nodes[i].output[0]: i for i in range(len(nodes))}
    # 初始化del_idxs，记录nodes所需修改
    # 0=不修改，1=删除，2=保留节点，设置输入shape和naame
    nodes_names = nodes_dict.keys()
    del_idxs = [0] * len(nodes_names)
    # 遍历需要去除的input_nodes，将他们的状态写到del_idxs
    for input_node_name in input_nodes_names:
        assert input_node_name in nodes_names, 'input node name %f is not in input onnx model' % input_node_name
        input_node_idx = nodes_dict[input_node_name]
        del_idxs[input_node_idx] = 2
        for node in nodes[input_node_idx::-1]:
            for node_input in node.input:
                if node_input in nodes_outputs_dict:
                    if del_idxs[input_node_idx] == 2 or del_idxs[input_node_idx] == 1:
                        del_idxs[nodes_outputs_dict[node_input]] = 1

    # 反向遍历所有nodes，将del_idxs标明需要修改的nodes按状态进行修改
    for i in range(len(del_idxs) - 1, -1, -1):
        if del_idxs[i] == 0:
            continue
        elif del_idxs[i] == 1:
            # 删掉input_nodes前的所有节点
            nodes.remove(nodes[i])
        elif del_idxs[i] == 2:
            # 记录input_nodes的output shape，然后设置新input name和tensor
            node_out_dim_info = in_graph.value_info[i-graph_input_num].type.tensor_type.shape.dim
            input_shape = [n.dim_value for n in node_out_dim_info]
            input_name = next(input_names_iter)
            # 给新的input node设定输入graph.input的name和tensor
            in_graph = add_node_inputs(in_graph, node_idx=i, input_name=input_name, input_shape=input_shape)

    new_nodes_inputs = [nodes[i].input[0] for i in range(len(nodes))]
    for inp in in_graph.input:
        if inp.name not in new_nodes_inputs:
            in_graph.input.remove(inp)

    return in_graph


def cut_output_nodes(in_graph, output_nodes_names=None):
    nodes = in_graph.node
    # 将node name与idx对应
    nodes_dict = {nodes[i].name: i for i in range(len(nodes))}
    # 将node output与idx对应
    nodes_outputs_dict = {nodes[i].output[0]: i for i in range(len(nodes))}
    # 初始化del_idxs，记录nodes所需修改
    # 0=不修改，1=删除，2=保留节点
    nodes_names = nodes_dict.keys()
    del_idxs = [0] * len(nodes_names)
    # 遍历需要去除的output_nodes，将他们的状态写到del_idxs
    for output_node_name in output_nodes_names:
        assert output_node_name in nodes_names, 'input node name %f is not in input onnx model' % output_node_name
        output_node_idx = nodes_dict[output_node_name]
        del_idxs[output_node_idx] = 2
        for node in nodes[output_node_idx:]:
            for node_input in node.input:
                if node_input in nodes_outputs_dict:
                    if del_idxs[nodes_outputs_dict[node_input]] == 2 or del_idxs[nodes_outputs_dict[node_input]] == 1:
                        del_idxs[nodes_dict[node.name]] = 1

    # 反向遍历所有nodes，将del_idxs标明需要修改的nodes按状态进行修改
    for i in range(len(del_idxs) - 1, -1, -1):
        if del_idxs[i] == 0:
            continue
        elif del_idxs[i] == 1:
            # 删掉output_nodes后的所有节点
            nodes.remove(nodes[i])

    new_nodes_outputs = [nodes[i].output[0] for i in range(len(nodes))]
    for output in in_graph.output:
        if output.name not in new_nodes_outputs:
            in_graph.output.remove(output)

    return in_graph


if __name__ == '__main__':
    yolox_opt = True
    print('========== Step 0. Load the onnx model ========')
    model = onnx.load(ONNX_MODEL_PATH)
    print('Done')
    print('========== Step 1. Simplify the onnx model ========')
    model_sim, ret = onnxsim.simplify(model)
    assert ret, 'Failed on simplifying the model'
    print('Done')
    graph = model_sim.graph
    print('========== Step 2. Remove the input nodes and add inputs ========')
    new_graph = cut_input_nodes(graph,
                                input_nodes_names=['Conv_41'],
                                input_names=['img_input'])
    print('Done')
    print('========== Step 3. Remove the output nodes and add outputs ========')
    if not yolox_opt:
        new_graph = cut_output_nodes(new_graph,
                                     output_nodes_names=['Transpose_570'])
    print('Done')
    print('========== Step 4. Save graph ========')
    onnx.checker.check_model(model_sim)
    onnx.save(model_sim, ONNX_OPT_SAVE_PATH)
    print('Done')
