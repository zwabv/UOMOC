# UOMOC
## a method for underwater image classification，directly related to the manuscript submitted to The Visual Computer.

This project leverages the CLIP pre-trained model, integrating adapters with learnable prompt learning strategies to accomplish multi-label classification tasks for underwater optical images. It supports both full-data training and few-shot learning modes, completing classification and evaluation for 10 categories of underwater objects. The project outputs multi-dimensional metrics including mAP, F1, and CNKI, while also providing resource assessments such as model parameter counts and computational requirements.
***
## Environment Preparation

### System Requirements

• Operating System: Linux/Windows/macOS (Linux recommended for GPU training)
• Python Version: 3.8 or higher
• CUDA Version: 11.3 or higher (recommended; CPU can run but training efficiency is extremely low)
• GPU Memory Requirement: ≥8GB (12GB or higher recommended for ViT-B/32 model compatibility)

### Dependency Installation.

Install dependencies
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
***
## Dataset Preparation

Organize image files and annotation files according to the following structure, placing them in the same directory as the main code file main.py:
├── main.py
├── images
│   ├── img_001.jpg
│   ├── img_002.png
│   └── ...
├── multilabel_train.json
├── multilabel_val.json
└── multilabel_test.json
***

## Code Execution
### Key Algorithms and Implementation
The core algorithm relies on "semantic refinement-cross-modal fusion-scale adaptation": it uses a category-related weight modulation algorithm to adaptively adjust textual features, constructs an image region-category semantics association model via cross-attention (capturing global semantics and long-range dependencies), and enhances the Transformer decoder’s FFN structure with a gating mechanism and receptive field expansion for multi-scale representation. Built on CLIP encoders, the workflow optimizes text features, fuses them with visual features, feeds the result into the modified decoder for refinement, and outputs multi-label classification via global pooling and activation—efficiently addressing complex underwater scenario demands.
### Training Commands
Within an activated virtual environment, navigate to the code root directory and execute the following commands. Supports command-line arguments for specifying few-shot parameters:

   (1) Single-round few-shot training
Specify the number of positive samples per class, applicable for FEWSHOT=True mode:
Example: 8 shots few-shot learning
python main.py --shots 8

   (2) Multi-round Few-Shot Training
Automatically runs five experimental groups with 1/2/4/8/16 shots, applicable to FEWSHOT=True mode:
python main.py --all_shots
***
## Acknowledgments and Citations
1. The foundational model is based on OpenAI's CLIP implementation;
2. Prompt learning strategies draw inspiration from CoOp and CoCoOp;
3. Traditional methods reference the original DistrEn paper's code.
     We express our deepest gratitude! We acknowledge the support provided by the public datasets RUOD, DUO, and MA-COCO.
     This project is intended solely for academic research. Should you utilize this code in your research, please cite the manuscript submitted to The Visual Computer.
