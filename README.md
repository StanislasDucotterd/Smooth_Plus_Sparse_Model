# Learning of Patch-Based Smooth-Plus-Sparse Models for Image Reconstruction
Implementation of experiments done in : 

![alt text](https://github.com/StanislasDucotterd/Smooth_Plus_Sparse_Model/blob/main/plot/decomposition.png?raw=true)

#### Description
We aim at the solution of inverse problems in imaging, by combining a penalized sparse representation of image patches with an unconstrained smooth one. This allows for a straightforward interpretation of the reconstruction.
We formulate the optimization as a bilevel problem.
The inner problem deploys classical algorithms while the outer problem optimizes the dictionary and the regularizer parameters through supervised learning.
The process is carried out via implicit differentiation and gradient-based optimization. 
We evaluate our method for denoising, super-resolution, and compressed-sensing magnetic-resonance imaging. We compare it to other classical models as well as deep-learning-based methods and show that it always outperforms the former and also the latter in some instances. 

#### Requirements
The required packages:
- `pytorch`
- `torchdeq`
- `numpy`
- `matplotlib` 
- `skimage`

#### Training

You can train a model with the following command:

```bash
python train.py --device cpu or cuda:n
```

#### Config file details️

Information about the hyperparameters that yield the best performance for the four experiments can be found in the config folder. 

Below we detail the model informations that can be controlled in the file `config.json`.

```javascript
{
    "dict_params": {
        "atom_size": 13,
        "beta_init": 2.0,
        "b_max_iter": 50,   // Hyperparameter of the backward solver for the DEQ
        "b_solver": "broyden",  // Hyperparameter of the backward solver for the DEQ
        "nb_atoms": 200,
        "nb_free_atoms": 120,
        "tol": 0.0001
    },
    "exp_name": "NCPR",
    "log_dir": "trained_models/sigma_25",
    "prox_params": {
        "groupsize": 1, // Refers to group sparsity, only used with l1norm
        "lambda_init": 0.0125,      
        "order": 1.8,   // Refers to the gamma of the nonconvex prox in the paper
        "prox_type": "l0norm"   //l1norm for convex and l0norm for nonconvex
    },
    "sigma": 25,    //noise level
    "training_options": {
        "batch_size": 16,
        "epochs": 1,
        "lr_atoms": 0.0002,
        "lr_decay": 0.9,
        "lr_hyper": 0.001,
        "num_workers": 1,
        "train_data_file": "/path/to/train.h5",
        "val_data_file": "/path/to/val.h5"
    }
}
```