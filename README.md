# Unlocking the matrix form of the Quaternion Fourier Transform and Quaternion Convolution: Properties, connections, and application to Lipschitz constant bounding

This is the official implementation for the theoretical tools introduced and discussed in the paper: *"Unlocking the matrix form of the Quaternion Fourier Transform and Quaternion Convolution: Properties, connections, and application to Lipschitz constant bounding"* (accepted in the **Transacations on Machine Learning Research Journal (TMLR)**, 2025)

[![OpenReview](https://img.shields.io/badge/OpenReview-View-brightgreen.svg)](https://openreview.net/forum?id=rhcpXTxb8j)
[![arXiv](https://img.shields.io/badge/arXiv-2307.01836-B31B1B.svg)](https://arxiv.org/abs/2307.01836)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
***

## Paper TL;DR

We show how the relation of convolution, circulant matrices and the Fourier transform generalizes to the quaternionic domain.
A Lipschitz constant bounding application acts as a proof-of-concept of the usefulness of our results.

## Paper Abstract

Linear transformations are ubiquitous in machine learning, and matrices are the standard way to represent them.
In this paper, we study matrix forms of quaternionic versions of the Fourier Transform and Convolution operations.
Quaternions offer a powerful representation unit, however they are related to difficulties in their use that stem foremost from non-commutativity of quaternion multiplication, 
and due to that $\mu^2 = -1$ possesses infinite solutions in the quaternion domain.
Handling of quaternionic matrices is consequently complicated in several aspects (definition of eigenstructure, determinant, etc.).
Our research findings clarify the relation of the Quaternion Fourier Transform matrix to the standard (complex) Discrete Fourier Transform matrix,
and the extend on which well-known complex-domain theorems extend to quaternions.
We focus especially on the relation of Quaternion Fourier Transform matrices to Quaternion Circulant matrices (representing quaternionic convolution), and the eigenstructure of the latter.
In the paper, a proof-of-concept application that makes direct use of our theoretical results is presented,
where we present a method to bound the Lipschitz constant of a Quaternionic Convolutional Neural Network.


## 💡 Project Overview

The code provides:
* An extensive **Python library** for working with quaternion matrix **eigenstructure**, the **quaternion Fourier transform**, create, visualize and manipulate **quaternion circulant matrices**.
* **Jupyter Notebooks** that demonstrate and numerically validate the main theoretical propositions presented in the paper.
* Routines to efficiently **evaluate the spectral norm** of a quaternion convolution operation, apply **clipping** and **Lipschitz constant bounding**.

The code makes use of the [quaternion numpy library](https://github.com/moble/quaternion).
***

## 📚 Repository Contents

The repository is structured to separate the core utilities from the demonstration notebooks:

| File/Folder | Description |
| :--- | :--- |
| `quaternion_matrix.py` | Core class for quaternion matrix algebra. |
| `quaternion_circulant_matrix.py` | Implementation of the Quaternion Circulant Matrix. |
| `quaternion_symplectic.py` | Implementation of quaternion symplectic operations. |
| `circulant.py` | Utility functions for standard circulant matrices. |
| `requirements.txt` | Python dependencies for setting up the environment. |
| `QFourier Notebook 01-07...ipynb` | **Demonstration Notebooks** numerical validation and demonstrators of the paper's propositions. |

### Key Demonstrations (Notebooks)

The following Jupyter Notebooks demonstrate the key theoretical results of the paper:

| Notebook | Focus/Proposition |
| :--- | :--- |
| `QFourier Notebook 03...` | **Proposition 3.1 & 3.2:** Showing the QFT as a matrix product. |
| `QFourier Notebook 04...` | **Proposition 3.3:** Demonstrating the convolution-multiplication property. |
| `QFourier Notebook 05...` | **Proposition 3.5:** Exploring the relationship between QCM and standard circulant matrices. |
| `QFourier Notebook 06...` | **Proposition 3.4:** Validating the spectral decomposition of the QCM. |
| `QFourier Notebook 07...` | Exploring properties like sums and products of circulant matrices. |

***

## ⚙️ Installation and Setup

To clone the repository and set up the necessary Python environment, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/sfikas/quaternion-fourier-convolution-matrix.git](https://github.com/sfikas/quaternion-fourier-convolution-matrix.git)
    cd quaternion-fourier-convolution-matrix
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip3 install -r requirements.txt
    ```

***

## 🏃 Getting Started (Usage)

All theoretical validations and usage examples are demonstrated within the **Jupyter Notebooks**.

1.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
2.  Navigate to the project directory in your browser.
3.  Open any of the `QFourier Notebook XX...ipynb` files to review the code and run the validations. The notebooks can be executed sequentially to follow the paper's flow.

***

## 📜 Citation

If you use this code or the concepts from the paper in your research, please cite the original work:

```bibtex
@article{sfikas2025unlocking,
  title={Unlocking the matrix form of the Quaternion Fourier Transform and Quaternion Convolution: Properties, connections, and application to Lipschitz constant bounding},
  author={Sfikas, Giorgos and Retsinas, George},
  journal={Transactions on Machine Learning Research Journal (TMLR)},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on GitHub.

## License

This project is licensed under the MIT License.