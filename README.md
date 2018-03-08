# Diagonalizing the Jaynes-Cummings ladder for cQED measurement of superconducting qubits

This code calculates the eigensystem of the Jaynes-Cummings ladder of a multi-level qubit (e.g., [transmon](https://arxiv.org/abs/cond-mat/0703002)) and its readout resonator. The resulted eigenenergies can be used to calculate interesting quantities that are usefull for understanding the dispersive measurement.

For a detailed discussion of the system, parameters and meaning of jargons used here see [Phys. Rev. A **94**, 012347 (2016)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.94.012347) and [Phys. Rev. Lett. **117**, 190503 (2016)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503). If you use this code in your project, please consider citing these papers.

Please refer to `examples.ipynb` jupyter notebook to see how the code can be used.

### Notes

* Code uses `python3.5.2`
* Was written with `numpy 1.12.1`. Other standard library used is `math`.
