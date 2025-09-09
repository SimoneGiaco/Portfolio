Jupyter Notebook with the code in Pytorch for a 'PINN' (Physics Informed Neural Network) to find numeric solutions of functional/differential equations. The model was trained on a T4 GPU in Google Colab. \
We solve a simple second order ordinary differential equation.\
The loss is constructed combining regularization terms which enforce the boundary conditions and the MSE loss for the equation on a grid of equally spaced points in the domain.
