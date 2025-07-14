"""
Utility file to convert the model into lower precision model format for deployment.
"""

import os
from typing import Any
import tensorrt as trt
import torch

def convert_model_to_low_precision(
    config: Any, model: torch.nn.Module, DEVICE: torch.device
) -> None:
    """
    Convert the trained model to low-precision formats (ONNX and TensorRT).

    Args:
        config (DictConfig): Configuration parameters.
        model (torch.nn.Module): The trained model to convert.
        DEVICE (torch.device): The device the model is currently on.
    """
    try:
        # Ensure the model is in evaluation mode
        model.eval()

        # Create output directory if it doesn't exist
        output_dir = config.output_folder_path
        os.makedirs(output_dir, exist_ok=True)

        # Export model to ONNX
        onnx_path = os.path.join(output_dir, 'model.onnx')
        print(f'Exporting model to ONNX format at {onnx_path}')

        # Simply a placeholder input for the model, 331090 is the max number of points in the dataset
        dummy_input = [torch.randn([331090, 3]).to(DEVICE)]
        dummy_input_image = [torch.randn([3, 565, 586]).to(DEVICE)]
        pcd_dims_min = [torch.tensor([0.0, 0.0, 0.0]).to(DEVICE)]
        pcd_dims_max = [torch.tensor([1.0, 1.0, 1.0]).to(DEVICE)]
        torch.onnx.export(
            model,
            (dummy_input, dummy_input_image, pcd_dims_min, pcd_dims_max),
            onnx_path,
            verbose=True,
            input_names=['input', 'input_image', 'pcd_dims_min', 'pcd_dims_max'],
            output_names=['output'],
            opset_version=16,
        )

        # Convert ONNX to TensorRT
        print('Converting ONNX to TensorRT...')
        trt_path = os.path.join(output_dir, 'model_trt.pth')

        # Use TensorRT's ONNX parser
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with (
            trt.Builder(TRT_LOGGER) as builder,
            builder.create_network() as network,
            trt.OnnxParser(network, TRT_LOGGER) as parser,
        ):
            # Set builder configuration
            builder.max_batch_size = 1
            builder.max_workspace_size = 1 << 30
            builder.fp16_mode = True  # Enable FP16 precision

            # Parse the ONNX model
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f'ONNX file {onnx_path} not found.')

            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in parser.get_errors():
                        print(f'ONNX Parser Error: {error}')
                    raise RuntimeError('Failed to parse ONNX file.')

            # Build the TensorRT engine
            engine = builder.build_cuda_engine(network)

            # Serialize the engine
            with open(trt_path, 'wb') as trt_file:
                trt_file.write(engine.serialize())

            print(f'TensorRT model saved at {trt_path}')

    except Exception as e:
        print(f'Error during model conversion: {str(e)}')
        raise
