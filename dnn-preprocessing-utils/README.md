# DNN Preprocessing Utilities

Python utilities for handling model preprocessing parameters and managing paths for various deep learning models.

## Features
- Extract preprocessing arguments from model zoo YAML files
- Automatically handle defaults for mean, scale, input size, and color order
- Supports multiple frameworks: Caffe, TensorFlow, Darknet, Torch, OpenVINO
- Resolve file paths in OpenCV test/data directories

## Requirements
```bash
pip install -r requirements.txt
