# Nonlinear-near-field-radiation-between-two-dipoles
This repository shows the code we used to calculate linear and nonlinear near field radiation between dipoles

The code is used for the hypothetic dipoles whose static (hyper)polarizability tensor only contains non-zero primary component (xx or xxx). To modify it for the real molecule, edit on the \alpha and \beta matrices by using the refs A. Schweig, Chem. Phys. Lett. 1, 163 (1967) and A. Schweig, Chem. Phys. Lett. 1, 195 (1967)

.nb file is run by Mathematica via .m file is run by the VASP with the run.sh file 

For the hypothetic dipole the resonant frequency is 2.9 eV, while the scattering rate (\eta) is taken as an empirical value (0.015*resonant)

Rotational symmetry is also considered and the rotational angle can be adapted in the code as well

The code of heat map generates integL12_XYZ.csv and integNL12_XYZ files that can be used to plot the linear and nonlinear heat maps in Originlab

The code of Q_vs_R_and_T generates Q_vs_R_Tsweep_series and Q_vs_T2_series that can be used to plot linear and nonlinear Q_vs_R, Q_vs_T(at R=10, 15A) figures.
