import math
import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer, total_steps):
    if args.lr_scheduler == 'StepLR':
        return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    def lr_lambda_cosine(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress))) 

    def lr_lambda_linear(current_step):
        if current_step < args.warmup_steps:
            return float(current_step) / float(max(1, args.warmup_steps))
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.0, 1.0 - progress)

    if args.lr_scheduler == 'warmup_cosine':
        return LambdaLR(optimizer, lr_lambda=lr_lambda_cosine)
    elif args.lr_scheduler == 'warmup_linear':
        return LambdaLR(optimizer, lr_lambda=lr_lambda_linear)