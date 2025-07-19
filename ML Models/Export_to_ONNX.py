import pandas as pd
from model_training import testing_dataloader
from CNN_Extended import CNNFingerprinter
import torch
import os

model = CNNFingerprinter()
model.load_state_dict(torch.load('RF_Model_Weights.pth'))
model.eval()

dummy_input = torch.randn(1, 2, 128)

torch.onnx.export(
    model, 
    dummy_input,
    "cnn_rf_model_128.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11 #Safe for onnx.js
)

print("Exported")


