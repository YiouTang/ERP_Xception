# ERP-Xception: Enhancing EEG Signal Classification with Channel Attention and Depthwise Separable Convolutions

## Overview
This project introduces the ERP-Xception model, a novel architecture that integrates channel attention mechanisms with depthwise separable convolutions. The model innovatively combines an enhanced ECA (Efficient Channel Attention) module with an optimized Xception structure and incorporates the ShiftVIT feature translation mechanism. The goal is to achieve high-quality feature extraction that captures the inter-channel interactions and temporal domain position information of EEG signals. Empirical results on the public ERP-CORE dataset demonstrate the model's efficient classification performance across six types of EEG signal components.
![替代文本](https://github.com/YiouTang/ERP_Xception/blob/main/images/git_title.png?raw=true)
## Project Structure
The project is organized into three main directories:

### Data_Progress
This folder contains essential data processing methods tailored for the ERP-core dataset, which consists of six sub-datasets. The preprocessing steps include calculating DTW (Dynamic Time Warping) distance matrices and generating Soft-DTW averaged data. For detailed implementation, refer to the publication:
> Ma Y, Tang Y, Zeng Y, Ding T, Liu Y (2023). An N400 identification method based on the combination of Soft-DTW and transformer. Front. Comput. Neurosci. 17:1120566. doi: 10.3389/fncom.2023.1120566.

Please note that the binary label files (`bini` and `binilabel`) provided with the ERP-core dataset are used for two types of label classifications. Except for the N2PC and N170 data, all other data are segmented using the `bini` labels. For specific label segmentation details, visit the official dataset website: [ERP-CORE](https://erpinfo.org/erp-core).

### Models
This directory houses the core data classification models of the project, implemented using PyTorch.

### Trained_Models
This folder contains the trained optimal model parameters and records of training results. The `.pth` files are PyTorch-readable model parameters.

## Usage
To utilize the models and data processing scripts, follow the instructions provided in the respective directories. Ensure that you have the necessary dependencies installed to run the Python scripts and PyTorch models.

## Contribution
Contributions to the project are welcome. Please adhere to the guidelines provided in the `CONTRIBUTING.md` file.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
