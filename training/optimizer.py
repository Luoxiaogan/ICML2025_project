# training/optimizer.py

import torch
from torch.optim import Optimizer

class PullDiag_GT(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store previous parameters and gradients as vectors
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()])
            .detach()
            .clone()
            for model in self.model_list
        ]

        # Initialize v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        # Initialize w_vector and prev_w_vector
        self.w_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )
        self.prev_w_vector = self.w_vector.clone()

        defaults = dict(lr=lr)
        super(PullDiag_GT, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        with torch.no_grad():
            # Update w_vector
            self.w_vector = torch.matmul(self.A, self.w_vector)

            # Step1: x = Ax
            prev_params_tensor = torch.stack(self.prev_params)  # (n_models, param_size)
            new_params_tensor = torch.matmul(
                self.A, prev_params_tensor
            )  # (n_models, param_size)

            # Step2: x = x - lr * v
            v_tensor = torch.stack(self.v_list)  # (n_models, param_size)
            new_params_tensor -= self.lr * v_tensor

            # Update model parameters
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        # Step3: compute new gradients
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # Compute new gradients
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)

            # Step4: v = A v + (1 / w_vector) * g - (1 / prev_w_vector) * prev_g
            v_tensor = torch.stack(self.v_list)
            prev_grads_tensor = torch.stack(self.prev_grads)

            weighted_v = torch.matmul(self.A, v_tensor)
            w_vector_inv = 1.0 / self.w_vector.unsqueeze(1)  # Shape (n_models, 1)
            prev_w_vector_inv = 1.0 / self.prev_w_vector.unsqueeze(1)

            W_g = w_vector_inv * new_grads_tensor
            prev_W_prev_g = prev_w_vector_inv * prev_grads_tensor
            new_v_tensor = weighted_v + W_g - prev_W_prev_g

            # Update v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # Step5: Update prev_params, prev_grads, prev_w_vector
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_w_vector = self.w_vector.clone()

        return loss
    
class PullDiag_GD(Optimizer):
    def __init__(self, model_list, lr=1e-2, A=None, closure=None):
        self.model_list = model_list
        self.lr = lr
        self.A = A.to(next(model_list[0].parameters()).device)

        # Compute initial gradients
        closure()

        # Store previous parameters and gradients as vectors
        self.prev_params = [
            torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
            for model in self.model_list
        ]
        self.prev_grads = [
            torch.nn.utils.parameters_to_vector([p.grad for p in model.parameters()])
            .detach()
            .clone()
            for model in self.model_list
        ]

        # Initialize v_list
        self.v_list = [prev_grad.clone() for prev_grad in self.prev_grads]

        # Initialize w_vector and prev_w_vector
        self.w_vector = torch.ones(
            self.A.shape[0], device=next(model_list[0].parameters()).device
        )
        self.prev_w_vector = self.w_vector.clone()

        defaults = dict(lr=lr)
        super(PullDiag_GD, self).__init__(model_list[0].parameters(), defaults)

    def step(self, closure, lr):
        self.lr = lr  # Update lr if provided

        with torch.no_grad():
            # Update w_vector
            self.w_vector = torch.matmul(self.A, self.w_vector)

            # Step1: x = Ax
            prev_params_tensor = torch.stack(self.prev_params)  # (n_models, param_size)
            new_params_tensor = torch.matmul(
                self.A, prev_params_tensor
            )  # (n_models, param_size)

            # Step2: x = x - lr * v
            v_tensor = torch.stack(self.v_list)  # (n_models, param_size)
            new_params_tensor -= self.lr * v_tensor

            # Update model parameters
            for i, model in enumerate(self.model_list):
                torch.nn.utils.vector_to_parameters(
                    new_params_tensor[i], model.parameters()
                )

        # Step3: compute new gradients
        for model in self.model_list:
            model.zero_grad()
        loss = closure()

        with torch.no_grad():
            # Compute new gradients
            new_grads = [
                torch.nn.utils.parameters_to_vector(
                    [p.grad for p in model.parameters()]
                )
                .detach()
                .clone()
                for model in self.model_list
            ]
            new_grads_tensor = torch.stack(new_grads)

            # Step4: v = A v + (1 / w_vector) * g - (1 / prev_w_vector) * prev_g
            v_tensor = torch.stack(self.v_list)
            prev_grads_tensor = torch.stack(self.prev_grads)

            weighted_v = torch.matmul(self.A, v_tensor)
            w_vector_inv = 1.0 / self.w_vector.unsqueeze(1)  # Shape (n_models, 1)
            prev_w_vector_inv = 1.0 / self.prev_w_vector.unsqueeze(1)

            W_g = w_vector_inv * new_grads_tensor
            prev_W_prev_g = prev_w_vector_inv * prev_grads_tensor
            new_v_tensor = weighted_v + W_g - prev_W_prev_g

            # Update v_list
            self.v_list = [new_v_tensor[i].clone() for i in range(len(self.model_list))]

            # Step5: Update prev_params, prev_grads, prev_w_vector
            self.prev_params = [
                new_params_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_grads = [
                new_grads_tensor[i].clone() for i in range(len(self.model_list))
            ]
            self.prev_w_vector = self.w_vector.clone()

        return loss