# M2ANet


A PyTorch implementation of M2ANet: Multi-Context Mask-Aware Attention Network for High-Fidelity Image Inpainting.

## Overview

Existing image inpainting methods have achieved promising performance using global attention mechanisms and generative adversarial networks. However, these methods face two key challenges: (1) Owing to the limited receptive field of traditional convolution, multi-scale feature fusion becomes incomplete, which leads to blurred details in complex irregular masks; (2) Adversarial training suffers from instability and mode collapse. To address these challenges, we propose M2ANet, a novel Transformer-based architecture that incorporates three key innovations: Firstly, a Depthwise Separable Mask-Aware Downsampling (DMD) module for robust feature and mask fusion. Secondly, a Dilated Convolution Block Attention Module (DCAM) is introduced to effectively synthesize multi-scale contextual information. Thirdly, we employ a contrastive learning enhanced adversarial training framework that stabilizes training and ensures semantic fidelity via multi-level feature constraints. Extensive experiments show that M2ANet achieves superior performance on the Paris Street View, CelebA-HQ, Dunhuang Challenge, and Dunhuang Mogao Grottoes murals datasets. On average across all datasets and mask ratios, it outperforms the second-best baseline by 6.1\% in FID and 10\% in LPIPS, demonstrating consistent and significant improvements in both perceptual and structural fidelity.

![fig2](https://github.com/user-attachments/assets/b3fafd1e-6cd8-4acf-8e94-f625a7f6d87b)


## Requirements

```bash
pip install torch torchvision
pip install opencv-python pillow
pip install wandb lpips thop
pip install scikit-image
pip install pyyaml
pip install numpy
```

## Project Structure

```
M2ANet/
├── src/
│   ├── M2ANet.py          # Main model implementation
│   ├── models.py           # Network architectures
│   ├── networks.py         # Neural network components
│   ├── loss.py             # Loss function implementations
│   ├── dataset.py          # Data loading and preprocessing
│   ├── config.py           # Configuration management
│   ├── metrics.py          # Evaluation metrics
│   ├── common.py           # Common utilities
│   └── utils.py            # Utility functions
├── checkpoint/             # Model checkpoints and config
├── utils/                  # Additional utilities
├── main.py                 # Main entry point
├── train.py                # Training script
├── test.py                 # Testing script
└── config.yml.example      # Configuration template
```

## Usage

### Training

```bash
python train.py
```

Or use the main script with training mode:
```bash
python main.py --path ./checkpoint
```

### Testing

```bash
python test.py --input path/to/images --mask path/to/masks --output path/to/results
```

Or use the main script with testing mode:
```bash
python main.py --path ./checkpoint --input path/to/images --mask path/to/masks --output path/to/results
```

### Configuration

The model configuration is managed through YAML files. Key parameters include:

- `MODE`: 1 for training, 2 for testing
- `MODEL`: Model type (2 for inpainting model)
- `MASK`: Mask type selection
- `BATCH_SIZE`: Training batch size
- `LR`: Learning rate
- `MAX_ITERS`: Maximum training iterations
- Loss weights for different components


## Dataset Preparation

Organize your data as follows:

```
dataset/
├── train_images/     # Training images
├── test_images/      # Test images
├── train_masks/      # Training masks
└── test_masks/       # Test masks
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation is based on and modified from the Edge-Connect framework.
