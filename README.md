# l1regKP
Codes for "Efficient Decomposition-Based Algorithms for ℓ1-Regularized Inverse Problems with Column-Orthogonal and Kronecker Product Matrices"

The Codes in the Figures folder reproduce the figures in the paper:
  Figures_Example1 produces the figures for Example 1, Figures_Example2 for Example 2, and Figures_Example3 for Example 3.

Codes in IterativeMethods run the SB and MM methods using the KP Structure:
  SB_GSVD2: Runs SB for provided lambda
  MM_GSVD2: Runs MM for provided lambda
  SB_ParamSel_GSVD2: Runs SB selecting lambda at each iteration using a provided selection method
  MM_ParamSel_GSVD2: Runs MM selecting lambda at each iteration using a provided selection method

Codes in ParameterSelection contain the codes for running the parameter selection methods: GCV, Central chi^2, and non-central chi^2.
Codes in RegularizationMatrix construct the framelet and D4 wavelet matrices used for regularization.
The true images for the examples are located in the ExampleImages folder.
