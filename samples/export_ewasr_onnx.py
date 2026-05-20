import sys, os
sys.path.insert(0, './eWaSR')
import torch
import onnx
import wasr.models as M
from wasr.utils import load_weights

WEIGHTS = './eWaSR/pretrained/ewasr_resnet18.pth'
OUTPUT  = './ewasr_resnet18.onnx'
SIZE    = (192, 256)  # H x W — our inference resolution

print("Loading model...")
model = M.get_model('ewasr_resnet18', num_classes=3, pretrained=False)
model.load_state_dict(load_weights(WEIGHTS))
model.eval()

# Dummy inputs at our inference resolution
dummy = {
    'image':    torch.randn(1, 3, SIZE[0], SIZE[1]),
    'imu_mask': torch.zeros(1, SIZE[0], SIZE[1])
}

print("Exporting to ONNX...")
torch.onnx.export(
    model,
    {'x': dummy},
    OUTPUT,
    opset_version=17,
    output_names=['prediction', 'intermediate'],
    do_constant_folding=True
)

# Verify
model_onnx = onnx.load(OUTPUT)
onnx.checker.check_model(model_onnx)
print(f"ONNX model valid: {OUTPUT}")
print(f"Size: {os.path.getsize(OUTPUT)/1024/1024:.1f} MB")
