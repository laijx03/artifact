import os
import argparse
import torch
import torch.nn as nn

# Input: [1, 32, 16, 16]
# Weight: [32, 32, 3, 3]
class Conv_32x32x16x3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x

def main(output_onnx):
    dummy_input = torch.randn(1, 32, 16, 16)
    torch.onnx.export(Conv_32x32x16x3(), (dummy_input), output_onnx, export_params=True, opset_version=10, verbose=False,
            input_names=['input'], output_names=['output'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate conv_32x32x16x3.onnx',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', default='./conv_32x32x16x3.onnx', help='Setting the output path')
    args = parser.parse_args()

    OUTPUT_ONNX = args.output
    OUTPUT_PATH = os.path.dirname(os.path.realpath(OUTPUT_ONNX))
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)

    main(OUTPUT_ONNX)
