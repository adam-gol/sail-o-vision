"""
TensorRT inference wrapper for eWaSR using PyTorch for buffer management.
No pycuda or cuda-python required.
"""
import numpy as np
import tensorrt as trt
import torch
import cv2

class EwasrTRT:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate GPU buffers as torch tensors
        self.buffers = {}
        self.input_names = []
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            # Map TRT dtype to torch dtype
            torch_dtype = torch.float16 if dtype == trt.float16 else torch.float32
            buf = torch.zeros(shape, dtype=torch_dtype, device='cuda').contiguous()
            self.buffers[name] = buf
            self.context.set_tensor_address(name, buf.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        print(f"TRT engine loaded")
        for n in self.input_names:
            print(f"  Input:  {n} {self.buffers[n].shape} {self.buffers[n].dtype}")
        for n in self.output_names:
            print(f"  Output: {n} {self.buffers[n].shape} {self.buffers[n].dtype}")

    def infer(self, image_np):
        """
        image_np: (H, W, 3) uint8 RGB numpy array
        Returns: (H, W) uint8 class mask (0=obstacle, 1=water, 2=sky)
        """
        img_name = self.input_names[0]
        _, C, H, W = self.buffers[img_name].shape

        # Preprocess image
        img = cv2.resize(image_np, (W, H), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / \
              np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_tensor = torch.from_numpy(
            img.transpose(2, 0, 1).reshape(1, C, H, W)
        ).to(dtype=torch.float32, device='cuda')

        self.buffers[img_name].copy_(img_tensor)

        # Run inference
        self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        # Get prediction — argmax over class dim
        pred = self.buffers[self.output_names[0]]  # (1, 3, H, W)
        mask = pred[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        return mask


if __name__ == '__main__':
    import time

    engine_path = './ewasr_resnet18.engine'
    print("Loading TRT engine...")
    model = EwasrTRT(engine_path)

    img = cv2.cvtColor(
        cv2.imread('./KOLOMVERSE/images/validation/0/0000000058.jpg'),
        cv2.COLOR_BGR2RGB)

    print("Warming up...")
    for _ in range(5):
        model.infer(img)

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        mask = model.infer(img)
        times.append((time.perf_counter()-t0)*1000)

    print(f"\nMean: {np.mean(times):.1f}ms  ({1000/np.mean(times):.0f} FPS)")
    print(f"Min:  {np.min(times):.1f}ms")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Obstacle px: {(mask==0).sum()}")
