from typing import cast, List, Optional, Tuple, Union

import torch

from torch import Tensor

from torch.optim.optimizer import _use_grad_for_differentiable, _get_value, _get_scalar_dtype

import torch.nn.functional as F


def _single_tensor_orthogonal_adamw(

    params: List[Tensor],

    grads: List[Tensor],

    exp_avgs: List[Tensor],

    exp_avg_sqs: List[Tensor],

    max_exp_avg_sqs: List[Tensor],

    state_steps: List[Tensor],

    grad_scale: Optional[Tensor],

    found_inf: Optional[Tensor],

    history_buffer_list: Optional[Tensor],

    *,

    amsgrad: bool,

    beta: float,

    orthogonal_eps: float,

    beta1: float,

    beta2: float,

    lr: Union[Tensor, float],

    weight_decay: float,

    eps: float,

    maximize: bool,

    capturable: bool,

    differentiable: bool,

):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():

        # this assert is due to JIT not realizing that the ops below

        # have overloads to handle both float and Tensor lrs, so we just assert it's

        # a float since most people using JIT are using floats

        assert isinstance(lr, float)

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]

        exp_avg = exp_avgs[i]

        exp_avg_sq = exp_avg_sqs[i]

        step_t = state_steps[i]

        # update step

        step_t += 1

        # Perform stepweight decay

        param.mul_(1 - lr * weight_decay)

        # for history buffer

        hbuf = history_buffer_list[i]

        if hbuf is None:

            hbuf = torch.clone(grad).detach()

            history_buffer_list[i] = hbuf

            lr_multiplier = 1

        else:

            # [out_channel, in_channel, *_]

            new_hbuf = torch.clone(grad).detach()

            # get grad A (past) and grad B (current), find component of B that is orthogonal to A

            # (A * B) / (|A|*|A|) * A

            # stable version: cos(A, B) * (|B| / |A|) * A

            cos_sim = F.cosine_similarity(
                new_hbuf, hbuf, dim=0, eps=orthogonal_eps
            ).unsqueeze(0)

            normalized_a = F.normalize(hbuf, dim=0, eps=orthogonal_eps)

            norm_b = torch.norm(new_hbuf, dim=0, keepdim=True)

            proj_b_on_a = cos_sim * norm_b * normalized_a

            grad = grad.add(proj_b_on_a, alpha=-1)

            lr_multiplier = 1

            hbuf.mul_(beta).add_(new_hbuf, alpha=1 - beta)

        # Decay the first and second moment running average coefficient

        exp_avg.lerp_(grad, 1 - beta1)

        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:

            step = step_t

            bias_correction1 = 1 - beta1**step

            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:

                # Maintains the maximum of all 2nd moment running avg. till now

                if differentiable:

                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()

                else:

                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient

                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write

                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)

                denom = (

                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)

                ).add_(eps / step_size_neg)

            else:

                denom = (

                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)

                ).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom, value=lr_multiplier)

        else:

            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step

            bias_correction2 = 1 - beta2**step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:

                # Maintains the maximum of all 2nd moment running avg. till now

                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient

                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)

            else:

                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            param.addcdiv_(exp_avg, denom, value=-step_size * lr_multiplier)


# The code below is copied & slightly modified from

# https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py

class OrthogonalAdamW(torch.optim.AdamW):

    def __init__(

        self,

        *args,

        orthogonal_beta: float = 0.9,

        orthogonal_eps: float = 1e-12,

        **kwargs,

    ):

        super().__init__(*args, **kwargs)

        for group in self.param_groups:

            group["orthogonal_beta"] = orthogonal_beta

            group["orthogonal_eps"] = orthogonal_eps

    def _init_group(

        self,

        group,

        params_with_grad,

        grads,

        amsgrad,

        exp_avgs,

        exp_avg_sqs,

        max_exp_avg_sqs,

        history_buffer_list,

        state_steps,

    ):

        has_complex = False

        for p in group["params"]:

            if p.grad is None:

                continue

            has_complex |= torch.is_complex(p)

            if has_complex:

                raise NotImplementedError

            params_with_grad.append(p)

            if p.grad.is_sparse:

                raise RuntimeError("AdamW does not support sparse gradients")

            grads.append(p.grad)

            state = self.state[p]

            # State initialization

            if len(state) == 0:

                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.

                # This is because kernel launches are costly on CUDA and XLA.

                state["step"] = (

                    torch.zeros(

                        (),

                        dtype=_get_scalar_dtype(is_fused=group["fused"]),

                        device=p.device,

                    )

                    if group["capturable"] or group["fused"]

                    else torch.tensor(0.0, dtype=_get_scalar_dtype())

                )

                # Exponential moving average of gradient values

                state["exp_avg"] = torch.zeros_like(

                    p, memory_format=torch.preserve_format

                )

                # Exponential moving average of squared gradient values

                state["exp_avg_sq"] = torch.zeros_like(

                    p, memory_format=torch.preserve_format

                )

                if amsgrad:

                    # Maintains max of all exp. moving avg. of sq. grad. values

                    state["max_exp_avg_sq"] = torch.zeros_like(

                        p, memory_format=torch.preserve_format

                    )

            exp_avgs.append(state["exp_avg"])

            exp_avg_sqs.append(state["exp_avg_sq"])

            if group["amsgrad"]:

                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            if group["differentiable"] and state["step"].requires_grad:

                raise RuntimeError(

                    "`requires_grad` is not supported for `step` in differentiable mode"

                )

            # Foreach without capturable does not support a tensor lr

            if (

                group["foreach"]

                and isinstance(group["lr"], Tensor)

                and not group["capturable"]

            ):

                raise RuntimeError(

                    "lr as a Tensor is not supported for capturable=False and foreach=True"

                )

            history_buffer_list.append(state.get("history_buffer"))

            state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.



        Args:

            closure (Callable, optional): A closure that reevaluates the model

                and returns the loss.

        """

        self._accelerator_graph_capture_health_check()

        loss = None

        if closure is not None:

            with torch.enable_grad():

                loss = closure()

        for group in self.param_groups:

            params_with_grad: List[Tensor] = []

            grads: List[Tensor] = []

            exp_avgs: List[Tensor] = []

            exp_avg_sqs: List[Tensor] = []

            max_exp_avg_sqs: List[Tensor] = []

            state_steps: List[Tensor] = []

            amsgrad: bool = group["amsgrad"]

            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            history_buffer_list: List[Optional[Tensor]] = []

            _ = self._init_group(

                group,

                params_with_grad,

                grads,

                amsgrad,

                exp_avgs,

                exp_avg_sqs,

                max_exp_avg_sqs,

                history_buffer_list,

                state_steps,

            )

            _single_tensor_orthogonal_adamw(

                params_with_grad,

                grads,

                exp_avgs,

                exp_avg_sqs,

                max_exp_avg_sqs,

                state_steps,

                history_buffer_list=history_buffer_list,

                amsgrad=amsgrad,

                beta=group["orthogonal_beta"],

                orthogonal_eps=group["orthogonal_eps"],

                beta1=beta1,

                beta2=beta2,

                lr=group["lr"],

                weight_decay=group["weight_decay"],

                eps=group["eps"],

                maximize=group["maximize"],

                capturable=group["capturable"],

                differentiable=group["differentiable"],

                grad_scale=getattr(self, "grad_scale", None),

                found_inf=getattr(self, "found_inf", None),

            )

            for p, history_buffer in zip(params_with_grad, history_buffer_list):

                state = self.state[p]

                state["history_buffer"] = history_buffer

        return loss
