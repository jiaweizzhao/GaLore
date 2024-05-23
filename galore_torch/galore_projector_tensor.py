import torch
from tensorly.decomposition import tucker
from tensorly import tenalg 

# The GaLoreProjector class in Python implements a projection method using orthogonal matrix
# decomposition for low-rank approximation of gradients for general tensors of dimension >2.
# We use tensor decomposition using tensorly library: https://tensorly.org/stable/index.html
class GaLoreProjectorTensor:
    """
    A class that represents a projector for the GaLore algorithm.

    Args:
        rank (int): The rank of the projector.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        update_proj_gap (int, optional): The number of iterations between updating the orthogonal matrix. Defaults to 200.
        scale (float, optional): The scaling factor for the projected gradients. Defaults to 1.0.
    """

    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.transformed_low_rank = None
        
    def project(self, full_rank_grad, iter):
        """
        Projects the full-rank gradients onto the low-rank subspace.

        Args:
            full_rank_grad (torch.Tensor): The full-rank gradients.
            iter (int): The current iteration.

        Returns:
            torch.Tensor: The transformed low-rank gradients.
        """
        if self.ortho_matrix is None and iter % self.update_proj_gap == 0:
            self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank)    
        self.transformed_low_rank = self.transform(self.ortho_matrix, full_rank_grad)
        return self.transformed_low_rank

    def project_back(self, low_rank_grad):
        """
        Projects the low-rank gradients back to the full-rank space.

        Args:
            low_rank_grad (torch.Tensor): The low-rank gradients.

        Returns:
            torch.Tensor: The full-rank gradients.
        """
        full_rank_grad = self.inverse_transform(self.ortho_matrix, self.transformed_low_rank)     
        return full_rank_grad * self.scale
        
    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank_all):
        """
        Computes the orthogonal matrix using SVD decomposition.

        Args:
            weights (torch.Tensor): The weights to decompose.
            rank_all (int): The desired rank of the decomposition.

        Returns:
            tuple: A tuple containing the core and factors of the orthogonal matrix.
        """
        module_params = weights
        if module_params.data.dtype != torch.float:
            matrix = module_params.data.float()
        else:
            matrix = module_params.data
        tucker_tensor = tucker(matrix, rank=rank_all)
        return tucker_tensor

    def transform(self, tensor, x):
        """
        Transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, tensor, x):
        """
        Inverse transforms the input tensor using the factors of the orthogonal matrix.

        Args:
            tensor (tuple): A tuple containing the core and factors of the orthogonal matrix.
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.
        """
        _, factors = tensor
        return tenalg.multi_mode_dot(x, factors)
