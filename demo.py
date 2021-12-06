import onnx
from onnxsim import simplify
onnx_model = onnx.load("/home/fandong/Code/model/yolov3_pruneV50_v1.onnx")  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, "yolov3-sim.onnx")
print('finished exporting onnx')
