from openvino.inference_engine import IECore


def openvino(openvino_weights_path):
    openvino_model_path=openvino_weights_path.replace('bin','xml')

    ie = IECore()
    net = ie.read_network(model=openvino_model_path, weights=openvino_weights_path)
    exec_net = ie.load_network(network=net, device_name="CPU")

    input_key = list(exec_net.input_info)[0]
    output_key = list(exec_net.outputs.keys())[0]
    network_input_shape = exec_net.input_info[input_key].tensor_desc.dims
    return exec_net, input_key
