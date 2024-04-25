import torch

def sketch_for_low_rank_approx_left(A, k, l, old_Q = False, Omega_old=None, old_Y = None):
    """
    Implement Algorithm 1: Sketch for Low-Rank Approximation
    
    Args:
        A (torch.Tensor): Input matrix of size (m, n)
        k (int): Sketch size parameter for range sketch
        l (int): Sketch size parameter for co-range sketch
        
    Returns:
        Omega (torch.Tensor): Random test matrix of size (n, k)
        Psi (torch.Tensor): Random test matrix of size (l, m)
        Y (torch.Tensor): Range sketch Y = AΩ of size (m, k)
        W (torch.Tensor): Co-range sketch W = ΨA of size (l, n)
    """
    m, n = A.size()
    original_type = A.data.dtype
    original_device = A.data.device
    theta = 1.0
    zeta = 1.0
    # Generate random test matrices
    if old_Q:
        Omega = Omega_old
        Y = theta * old_Y + zeta * A @ Omega
    else:
        Omega = torch.randn(n, k).to(original_device).type(original_type)
        Y = A @ Omega
    Psi = None
    W = None

    Q, _ = torch.linalg.qr(Y.type(torch.float32))
    Q = Q.type(torch.float32)
    return Omega, Psi, Y, W, Q

def sketch_for_low_rank_approx_right(A, k, l, old_Q = False, Psi_old=None, old_W = None):
    m, n = A.size()
    original_type = A.data.dtype
    original_device = A.data.device
    theta = 1.0
    zeta = 1.0
    # Generate random test matrices
    if old_Q:
        Psi = Psi_old
        W = theta * old_W + zeta * Psi @ A
    else:
        Psi = torch.randn(l, m).to(original_device).type(original_type)
        W = Psi @ A
    Omega = None
    Y = None

    Q, _ = torch.linalg.qr(W.T.type(torch.float32))
    Q = Q.type(torch.float32)
    return Omega, Psi, Y, W, Q

def low_rank_approx(Y, W, Psi):
    """
    Implement Algorithm 4: Low-Rank Approximation
    
    Args:
        Y (torch.Tensor): Range sketch Y = AΩ of size (m, k)
        W (torch.Tensor): Co-range sketch W = ΨA of size (l, n)
        Psi (torch.Tensor): Random test matrix Psi of size (l, m)
        
    Returns:
        Q (torch.Tensor): Orthonormal basis for range of Y of size (m, k)
        X (torch.Tensor): Factor matrix of size (k, n)
        A_approx (torch.Tensor): Low-rank approximation QX of size (m, n)
    """
    # Step 1: Form an orthogonal basis for the range of Y
    Q, _ = torch.linalg.qr(Y.type(torch.float32))
    Q = Q.type(torch.float32)
    Psi = Psi.type(torch.float32)
    W = W.type(torch.float32)
    # Step 2: Orthogonal-triangular factorization of ΨQ
    PsiQ = Psi @ Q
    U, T = torch.linalg.qr(PsiQ)
    
    # Step 3: Solve the least-squares problem to obtain X
    X = torch.linalg.lstsq(T, U.T @ W).solution
    
    # Step 4: Construct the rank-k approximation
    A_approx = Q @ X
    
    return Q, X, A_approx

class GaLoreProjectorSketching:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std'):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_type = proj_type
        self.Omega = None
        self.Psi = None
        self.Y = None
        self.W = None
        
    def project(self, full_rank_grad, iter):
        if self.Omega is None or self.Psi is None or iter % self.update_proj_gap == 0:
            if self.Omega is None or self.Psi is None:
                if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                    self.Omega, self.Psi, self.Y, self.W, self.Q = sketch_for_low_rank_approx_left(full_rank_grad, self.rank, self.rank, False, None, None)
                else:
                    self.Omega, self.Psi, self.Y, self.W, self.Q = sketch_for_low_rank_approx_right(full_rank_grad, self.rank, self.rank, False, None, None)
            else:
                if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                    self.Omega, self.Psi, self.Y, self.W, self.Q = sketch_for_low_rank_approx_left(full_rank_grad, self.rank, self.rank, True, self.Psi, self.Y)
                else:
                    self.Omega, self.Psi, self.Y, self.W, self.Q = sketch_for_low_rank_approx_right(full_rank_grad, self.rank, self.rank, True, self.Omega, self.W)
        original_device = full_rank_grad.device
        original_type = full_rank_grad.dtype
        self.Q = self.Q.to(original_device).type(original_type)
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            low_rank_grad = torch.matmul(self.Q.T, full_rank_grad)
        else:
            low_rank_grad = torch.matmul(full_rank_grad, self.Q)
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(self.Q, low_rank_grad.T).T
        else:
            full_rank_grad = torch.matmul(low_rank_grad.T, self.Q.T).T
        return full_rank_grad * self.scale