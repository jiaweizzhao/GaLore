import torch

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', proj_quant=False):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale

        self.ortho_matrix = None
        self.ortho_matrix_scales = None
        self.ortho_matrix_zeros = None
        self.ortho_matrix_shape = None

        self.proj_type = proj_type

        self.proj_quant = proj_quant
        self.quant_group_size = 256
        self.quant_n_bit = 4

    def project(self, full_rank_grad, iter):
        # TODO: implement the quantizated projection for other proj_type, currently only support std

        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')

                if self.proj_quant:
                    float_ortho_matrix = self.unpack_int4_projection()
                else:
                    float_ortho_matrix = self.ortho_matrix

                low_rank_grad = torch.matmul(full_rank_grad, float_ortho_matrix.t().to(full_rank_grad.device.type))
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')

                if self.proj_quant:
                    float_ortho_matrix = self.unpack_int4_projection()
                else:
                    float_ortho_matrix = self.ortho_matrix

                low_rank_grad = torch.matmul(float_ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)

        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type),full_rank_grad)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = torch.matmul(full_rank_grad,self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t().to(full_rank_grad.device.type))
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = torch.matmul(self.ortho_matrix.t().to(full_rank_grad.device.type), full_rank_grad)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            low_rank_grad = torch.matmul(self.ortho_matrix[0].t().to(full_rank_grad.device.type), full_rank_grad) @ self.ortho_matrix[1].t().to(full_rank_grad.device.type)

        return low_rank_grad

    def project_back(self, low_rank_grad):
        # TODO: implement the quantizated projection for other proj_type, currently only support std
        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                if self.proj_quant:
                    float_ortho_matrix = self.unpack_int4_projection()
                else:
                    float_ortho_matrix = self.ortho_matrix
                full_rank_grad = torch.matmul(low_rank_grad, float_ortho_matrix.to(low_rank_grad.device.type))
            else:
                if self.proj_quant:
                    float_ortho_matrix = self.unpack_int4_projection()
                else:
                    float_ortho_matrix = self.ortho_matrix
                full_rank_grad = torch.matmul(float_ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)

        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.to(low_rank_grad.device.type))
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix.to(low_rank_grad.device.type), low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0].to(low_rank_grad.device.type), low_rank_grad) @ self.ortho_matrix[1].to(low_rank_grad.device.type)


        return full_rank_grad * self.scale

    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights

        if module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        U, s, Vh = torch.linalg.svd(matrix, full_matrices = False)

        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            B = Vh[:rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)

            if self.proj_quant:
                self._quantize(B, q_group_size=self.quant_group_size, n_bit=self.quant_n_bit)
            else:
                self.ortho_matrix = B
            
            return B

        elif type=='left':
            A = U[:, :rank]
            if not float_data:
                A = A.to(original_device).type(original_type)

            if self.proj_quant:
                self._quantize(A, q_group_size=self.quant_group_size, n_bit=self.quant_n_bit)
            else:
                self.ortho_matrix = A

            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

    def _quantize(self, w, q_group_size=-1, n_bit=4):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)

        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int).to(torch.uint8)
        packed_w = self.pack_uint8_to_int4(w)

        self.ortho_matrix = packed_w
        self.ortho_matrix_scales = scales
        self.ortho_matrix_zeros = zeros
        self.ortho_matrix_shape = org_w_shape

    def pack_uint8_to_int4(self,tensor):
        reshaped = tensor.view(tensor.shape[0], -1, 2)
        packed = (reshaped[:, :, 0] & 0x0F) | ((reshaped[:, :, 1] & 0x0F) << 4)
        return packed

    def unpack_int4_projection(self):
        packed_tensor = self.ortho_matrix
        unpacked_low = packed_tensor & 0x0F
        unpacked_high = (packed_tensor >> 4) & 0x0F
        unpacked = torch.stack([unpacked_low, unpacked_high], dim=-1).view(packed_tensor.shape[0], -1)

        float_ortho_matrix = self.ortho_matrix_scales * (unpacked.to(self.ortho_matrix_scales.dtype) - self.ortho_matrix_zeros)
        float_ortho_matrix = float_ortho_matrix.reshape(self.ortho_matrix_shape)
        return float_ortho_matrix








