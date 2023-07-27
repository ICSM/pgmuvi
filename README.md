# pgmuvi
Python gaussian processes for multiwavelength variability inference

[![Documentation Status](https://readthedocs.org/projects/pgmuvi/badge/?version=latest)](https://pgmuvi.readthedocs.io/en/latest/?badge=latest)


pgmuvi is based on GPyTorch and intended for us in infering the properties of astronomical sources with multiwavelength variability. It uses spectral-mixture kernels to learn an approximation of the PSD of the variability, which have been shown to be very effective for pattern discovery (see https://arxiv.org/pdf/1302.4245.pdf). 
