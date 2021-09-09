import onnx
from onnx import shape_inference

model = onnx.load("model.onnx")

# shape inference for input
tensor = onnx.TensorProto()
with open("input_0.pb", 'rb') as f:
    tensor.ParseFromString(f.read())
g_input = model.graph.input[0]
g_input.type.tensor_type.shape.dim[0].dim_value = tensor.dims[0]
g_input.type.tensor_type.shape.dim[1].dim_value = tensor.dims[1]

inferred_model = shape_inference.infer_shapes(model)

for value in inferred_model.graph.value_info:
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            print(value.name)
            break
