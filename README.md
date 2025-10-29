# PINN m-Method for Pile Analysis

This repository contains code for a Physics-Informed Neural Network (PINN) implementation of the m-method for pile analysis, based on the work by Zexiong Wu and Xueyou Li. The code is provided in Jupyter notebooks for training and loading the model.

- **Authors**: Zexiong Wu (wuzx58@mail2.sysu.edu.cn), Xueyou Li (lixueyou@mail.sysu.edu.cn)
- **Date**: 2024-06-09
- **Description**: Implementation of PINN for the m-method in pile engineering.
- **Related Article**: [Link to the article](https://link.cnki.net/urlid/32.1124.TU.20250926.1432.002)

## Overview

The repository includes:
- `Train_model.ipynb`: Jupyter notebook for training the PINN model from scratch.
- `Load_model.ipynb`: Jupyter notebook for loading a pre-trained model and making predictions.
- `PINN_m_method_pile.pth`: Pre-trained model weights file.

The model uses PyTorch to define a custom neural network with hard constraints for physics-informed predictions. It computes physical quantities such as displacement (y), rotation (Θ), moment (M), shear force (Fs), and soil reaction (p) for a pile under given loads.

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

## Usage

### Training the Model
1. Open `Train_model.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Run the cells to define the network, set parameter ranges, and train the model.
3. The training loop optimizes the model using Adam optimizer and tracks losses (governing equation, boundary conditions, etc.).
4. After training, save the model weights (e.g., using `torch.save(model.state_dict(), 'PINN_m_method_pile.pth')`).

Example training output includes loss values for different components (gov, Mt, Vt, Mb, Vb) and a visualization of predictions.

### Loading and Predicting with the Pre-trained Model
1. Ensure `PINN_m_method_pile.pth` is in the same directory.
2. Open `Load_model.ipynb` in Jupyter Notebook or Jupyter Lab.
3. Run the cells to define the network and load the model using the `load_model` function:
   ```python
   def load_model(filename):
       model = CustomNetwork(layers=[4] + [50] * 4 + [2], activation=nn.Tanh(), is_TanhShrink=True)
       model.load_state_dict(torch.load(filename + '.pth'))
       model.eval()
       return model
   
   model = load_model('PINN_m_method_pile')
   ```
4. Provide input parameters (e.g., Lp, E, Vt, Mt, D, b0, m) and run the prediction section.
5. The notebook generates plots for the predicted physical quantities along the pile length.

Example inputs:
- Lp = 15 (pile length)
- E = 2.8e7 (modulus of elasticity)
- Vt = 60 (vertical load)
- Mt = 700 (moment load)
- D = 1.5 (pile diameter)
- b0 = 2.25 (soil parameter)
- m = 9400 (soil modulus)

The output includes a figure with 5 subplots showing y, Θ, M, Fs, and p vs. depth (x).

## Model Details

- **Network Architecture**: Feedforward neural network with 4 input features (normalized x, Lp, Mt, Vt), hidden layers of 50 neurons each (4 layers), and 2 outputs. Uses TanhShrink activation for the first and last hidden layers.
- **Normalization**: Inputs are normalized using α = (m * b0 / EI)^{1/5}.
- **Hard Constraints**: Applied in the forward pass to enforce boundary conditions.
- **Loss Function**: Mean squared error on the governing equation and boundary conditions (Mt, Vt, Mb, Vb).
- **Gradients**: Computed using PyTorch's autograd for higher-order derivatives (up to 4th order).
