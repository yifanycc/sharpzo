import os
import random
import argparse
import yaml
import time
import torchvision.transforms as transforms
from torch.utils.data import Subset

from datasets import build_dataset
from datasets.utils import build_data_loader
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import numpy as np

from utils import *
import cma

import gc
from functorch import vmap, jvp, jacrev, make_functional_with_buffers
import wandb
from datasets.imagenet import ImageNet
from collections import defaultdict
import pdb

_tokenizer = _Tokenizer()
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])


class ZOTrainer:
    def __init__(self, args, model, prompt_learner):
        self.args = args
        self.named_parameters_to_optim = []
        self.zo_random_seed = []
        self.projected_grad = []
        self.zo_random_seed = 0
        self.zo_eps_cge = args.zo_eps_cge
        self.zo_eps_rge = args.zo_eps_rge
        self.projected_grad = 0
        self.lr = args.zo_lr
        self.model = model
        self.prompt_learner = prompt_learner
        self.weight_decay = 0
        self.num_pertub = args.num_pertub
        self.grad_tmp = defaultdict(float)
        self.pruning_score = {}

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1, pruning_score=None):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            if self.pruning_score is not None and self.args.prompt_trainer == 'zo-pruned':
                k = int(self.args.prune_ratio * param.data.numel())
                score = self.pruning_score[name]  # shape: (e.g., [40, 512])
                mask = torch.zeros_like(score)
                sort_res = torch.sort(score, dim=-1, stable=True)
                indices = sort_res[1][:int(len(score) * self.args.prune_ratio)]
                mask.scatter_(0, indices, True)

            else:
                mask = 1.0

            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zo_eps_rge * mask

    def zo_forward(self, input=None, target=None, train_features=None, train_labels=None):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        self.model.eval()
        with torch.no_grad():
            prompts, tokenized_prompts = self.prompt_learner(trainer='zo')
            loss, _ = self.model(
                prompts=prompts,
                tokenized_prompts=tokenized_prompts,
                input=input,
                target=target,
                train_features=train_features,
                train_labels=train_labels,
            )
        return loss

    def zo_step_cge(self, input=None, target=None, train_features=None, train_labels=None, steps=None, mode='prompt'):
        # What parameters to optimize
        self.named_parameters_to_optim = []
        if mode == 'prompt':
            for name, param in self.prompt_learner.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((name, param))
        elif mode == 'model':
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.named_parameters_to_optim.append((name, param))
        else:
            raise NotImplementedError
        for name, param in self.named_parameters_to_optim:
            if 'input_param' in name:
                param = param.mean(dim=0)
        for name, param in self.named_parameters_to_optim:
            # param = param.mean(dim=0)
            grad = torch.zeros_like(param.data)
            if len(param.shape) == 1:
                for i in tqdm(range(self.args.intrinsic_dim)):
                    # idx = np.unravel_index(i, param.data.shape)
                    # First function evaluation
                    param.data[i] += self.zo_eps_cge
                    loss1 = self.zo_forward(input, target, train_features, train_labels)
                    # Second function evaluation
                    param.data[i] -= 2 * self.zo_eps_cge
                    loss2 = self.zo_forward(input, target, train_features, train_labels)
                    # Restore the parameter
                    param.data[i] += self.zo_eps_cge
                    # Compute the gradient
                    grad[i] = (loss1 - loss2) / (2 * self.zo_eps_cge)
            elif len(param.shape) == 2:
                for i in tqdm(range(param.data.shape[1])):
                    # idx = np.unravel_index(i, param.data.shape)
                    # First function evaluation
                    param.data[:, i] += self.zo_eps_cge
                    loss1 = self.zo_forward(input, target, train_features, train_labels)
                    # Second function evaluation
                    param.data[:, i] -= 2 * self.zo_eps_cge
                    loss2 = self.zo_forward(input, target, train_features, train_labels)
                    # Restore the parameter
                    param.data[:, i] += self.zo_eps_cge
                    # Compute the gradient
                    grad[:, i] = (loss1 - loss2) / (2 * self.zo_eps_cge)

            else:
                raise NotImplementedError
            self.grad_tmp[name] = grad

        return loss1

    def zo_step(self, input=None, target=None, train_features=None, train_labels=None, steps=None, mode='prompt'):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)

        args:
        mode: choose from 'prompt' or 'model' to optimize prompt or model parameters
        """

        projected_grad_list = []
        zo_random_seed_list = []
        # What parameters to optimize
        self.named_parameters_to_optim = []

        for name, param in self.prompt_learner.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        # update the pruning mask
        if self.args.prompt_trainer == 'zo-pruned' and (
                steps - self.args.change_point - 1) % self.args.prune_interval == 0:

            for name, param in self.named_parameters_to_optim:
                self.zo_step_cge(
                    train_features=train_features,
                    train_labels=train_labels,
                )
                param_grad = list(self.grad_tmp.items())[0][1]
                self.pruning_score[name] = self._pruner_zero_score(param, param_grad)

        for i in range(self.num_pertub):
            # Sample the random seed for sampling z
            # zo_random_seed = np.random.randint(1000000000)
            zo_random_seed = np.random.randint(1000000000)
            zo_random_seed_list.append(zo_random_seed)
            # First function evaluation
            self.zo_perturb_parameters(scaling_factor=1, random_seed=zo_random_seed)
            loss1 = self.zo_forward(input, target, train_features, train_labels)

            # Second function evaluation
            self.zo_perturb_parameters(scaling_factor=-2, random_seed=zo_random_seed)
            loss2 = self.zo_forward(input, target, train_features, train_labels)

            projected_grad = ((loss1 - loss2) / (2 * self.zo_eps_rge)).item()
            projected_grad_list.append(projected_grad)
            self.zo_perturb_parameters(scaling_factor=1, random_seed=zo_random_seed)
        self.zo_random_seed = zo_random_seed_list
        self.projected_grad = projected_grad_list

        # compute the exact gradient
        for i in range(self.num_pertub):
            torch.manual_seed(self.zo_random_seed[i])

            for name, param in self.named_parameters_to_optim:
                # Resample z
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                                 dtype=param.data.dtype)
                self.grad_tmp[name] += self.projected_grad[i] * z / self.num_pertub
        return loss1

    def _z_score_scale(self, tensor):
        mean = tensor.mean()
        std = tensor.std(unbiased=False)  # use unbiased=False to match numpy std behavior
        return (tensor - mean) / (std + 1e-8)  # avoid division by zero

    def _pruner_zero_score(self, param, param_grad):
        # Take absolute values
        abs_w = torch.abs(param)
        abs_g = torch.abs(param_grad)

        # Elementwise scaling of |G| using min-max normalization
        scaled_g = self._z_score_scale(abs_g)

        # Final pruning score: |W| * |W| * σ(|G|)
        score = abs_w * abs_w * scaled_g
        return score

    def zo_update(self):
        """
        Update the parameters with the estimated gradients.
        """

        for name, param in self.named_parameters_to_optim:
            if self.pruning_score is not None and self.args.prompt_trainer == 'zo-pruned':
                k = int(self.args.prune_ratio * param.data.numel())
                score = self.pruning_score[name]  # shape: (e.g., [40, 512])
                mask = torch.zeros_like(score)
                sort_res = torch.sort(score, dim=-1, stable=True)
                indices = sort_res[1][:int(len(score) * self.args.prune_ratio)]
                mask.scatter_(0, indices, True)

            else:
                mask = 1.0
            if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                param.data = param.data - self.lr * (
                        self.grad_tmp[name] + self.weight_decay * param.data) * mask
            else:
                param.data = param.data - self.lr * self.grad_tmp[name] * mask


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection.type(self.dtype)
        return x


class PromptLearner(nn.Module):
    def __init__(self, args, cfg, classnames, clip_model, val_loader, train_loader_F):
        super().__init__()
        self.device = torch.cuda.current_device()
        self.train_features, self.train_labels = pre_load_features(cfg, "train", clip_model, train_loader_F)
        n_cls = len(classnames)
        n_ctx = 4
        self.args = args
        self.cfg = cfg
        self.clip_model = clip_model
        ctx_init = 'a photo of a'
        self.dtype = clip_model.dtype
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :].cuda()
        prompt_prefix = ctx_init
        self.n_ctx = n_ctx

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = ctx_vectors
        self.bias = torch.rand_like(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, clip_model)
        self.linear = nn.Linear(cfg["intrinsic_dim"], n_ctx * ctx_vectors.shape[1], bias=False)
        self.input_param = torch.nn.Parameter(
            torch.rand((40, cfg["intrinsic_dim"]), dtype=self.linear.weight.dtype), requires_grad=True)

        for p in self.linear.parameters():
            torch.nn.init.normal_(p, 0, cfg["std"])

    def get_prefix_suffix_token(self, classnames, clip_model):
        prompt_prefix = self.prompt_prefix
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.text_encoder = TextEncoder(clip_model).float().cuda()
        self.name_lens = name_lens

    def forward(self, prompt_embedding=None, model=None, steps=None, trainer=None):
        # tunable head optimization.
        if prompt_embedding is None:
            if self.args.smoothness_estimator != None:
                ctx = self.ctx  # p_0
                ctx = ctx + self.linear(self.input_param).reshape(-1, self.n_ctx, self.ctx.shape[1]).mean(dim=0)
            else:
                ctx = self.ctx  # p_0
                ctx = ctx + self.bias.reshape(-1, self.n_ctx, self.ctx.shape[1]).mean(dim=0)
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            return prompts, self.tokenized_prompts

        # cma-es optimization.
        else:
            ctx = self.ctx  # p_0
            ctx = ctx + self.linear(prompt_embedding).reshape(self.n_ctx, -1)  # p_0 + Az
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            prefix = self.token_prefix
            suffix = self.token_suffix

            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
            with torch.no_grad():
                loss, _ = model(
                    prompts=prompts,
                    tokenized_prompts=self.tokenized_prompts,
                    train_features=self.train_features,
                    train_labels=self.train_labels,
                )

            return loss.unsqueeze(0)

    def update_context_prompt(self, es):
        solutions = es.ask()  # list of numpy array. [numpy.ndarray]. len(solutions) = cfg["popsize"]
        inputs = torch.tensor(np.array(solutions)).cuda().float()
        with torch.no_grad():
            self.input_param.copy_(inputs)
        self.bias = self.linear(inputs).reshape(-1, self.n_ctx, self.ctx.shape[1]).mean(dim=0)


class BBCLIP(nn.Module):
    def __init__(self, cfg, args, clip_model):
        super(BBCLIP, self).__init__()
        self.cfg = cfg
        self.args = args
        self.clip_model = clip_model
        self.text_encoder = TextEncoder(clip_model).float()

    def forward(self, prompts, tokenized_prompts, input=None, target=None, train_features=None, train_labels=None):
        text_features = self.text_encoder(prompts.float(), tokenized_prompts.float())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if train_features is not None:
            image_features = train_features
        else:
            image_features = self.clip_model.encode_image(input)
        if train_features is not None:
            assert train_labels is not None
            target = train_labels

        logits = 100. * image_features.float() @ text_features.T.float()

        update_logits = logits
        loss = F.cross_entropy(logits, target.long())

        return loss, update_logits


def run(args, cfg, dataset, clip_model, val_loader, test_loader, train_loader_F, run_name):
    device = torch.cuda.current_device()
    # Pre-load val features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader)
    prompt_learner = PromptLearner(args, cfg, dataset.classnames, clip_model, val_loader, train_loader_F).cuda()
    bb_clip = BBCLIP(cfg=cfg, args=args, clip_model=clip_model)

    optimizer = torch.optim.Adam(prompt_learner.parameters(), lr=args.zo_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_steps)

    # compute the kernel
    cfg['kernel_formula'] = 'signgd'
    cfg['binary_classification'] = False
    cfg['kernel_bs'] = 4

    num_samples = 8
    sampled_indices = random.sample(range(len(dataset.test)), num_samples)
    # Create a new dataset containing only the sampled indices
    shrunken_dataset = Subset(dataset.test, sampled_indices)

    # define the trainer

    cma_opts = {
        'seed': cfg["seed"],
        'popsize': cfg["popsize"],
        # 'maxiter': cfg["budget"] if cfg["parallel"] else cfg["budget"] // cfg["popsize"],
        'maxiter': 10000,
        'verbose': -1,
    }
    if cfg["bound"] > 0:
        cma_opts['bounds'] = [-1 * cfg["bound"], 1 * cfg["bound"]]
    es = cma.CMAEvolutionStrategy(cfg["intrinsic_dim"] * [0], args.sigma, inopts=cma_opts)
    print('Population Size: {}'.format(es.popsize))
    print('{} Evaluation.'.format('Parallel' if cfg["parallel"] else 'Serial'))
    zo_trainer = ZOTrainer(args=args, model=bb_clip, prompt_learner=prompt_learner)

    print(f'Printing trainable parameters')
    for name, param in prompt_learner.named_parameters():
        if name == 'input_param':
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in clip_model.named_parameters():
        param.requires_grad = False

    print(f'Printing to verify trainable parameters')
    for name, param in prompt_learner.named_parameters():
        if param.requires_grad:
            print(f'name {name} param.shape: {param.shape}')
    for name, param in clip_model.named_parameters():
        if param.requires_grad:
            print(f'name {name} param.shape: {param.shape}')

    start_step = 0
    loss = 0
    start_stage2 = False
    best_acc = 0
    delta = 0.001
    patience = 10
    patience_counter = 0
    # beta: hybrid cma-es-zo method
    for step in range(start_step, args.total_steps):

        if step < args.change_point:
            solutions = es.ask()  # list of numpy array. [numpy.ndarray]. len(solutions) = cfg["popsize"]
            inputs = torch.tensor(np.array(solutions)).cuda().float()
            eps = compute_eps(args, bb_clip, prompt_learner, zo_trainer)
            if args.smoothness_estimator != 'none':
                eps = eps.mean(dim=0)
            losses = [prompt_learner(x + eps, steps=step, model=bb_clip, trainer='cma-es') for x in inputs]
            fitnesses = [loss.item() for loss in losses]
            es.tell(solutions, fitnesses)
            # es.disp()
            prompt_learner.update_context_prompt(es)
            scheduler.step()
            evaluation_period = 5

        else:
            if step == args.change_point:
                print(f'merge the parameter at step {step}')
                # Compress CMA-ES solution (40 x D) to a single vector (1 x D)
                with torch.no_grad():
                    compressed = prompt_learner.input_param.detach().mean(dim=0)  # (D,)

                prompt_learner.register_parameter(
                    "input_param",
                    torch.nn.Parameter(compressed.clone(), requires_grad=True)
                )
            loss = zo_trainer.zo_step(
                train_features=prompt_learner.train_features,
                train_labels=prompt_learner.train_labels,
                steps=step,
            )
            zo_trainer.zo_update()
            scheduler.step()
            evaluation_period = 100
        if step % evaluation_period == 0:
            acc = evaluate_model(args, bb_clip, prompt_learner, test_features, test_labels)
            wandb.log({"step": step, "acc": acc, "loss": loss})
            print("Steps: %s \t Acc:%s \t Loss:%s" % (step, acc, loss))
    if args.save_model:
        checkpoint_dir = f'{args.root_path}/mllm/checkpoints_auto'
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_name}_checkpoint_step_{change_points_auto}.pt")
        torch.save({
            'prompt_learner_state': prompt_learner.state_dict(),
            'es_state': es.result,  # or use a custom serialization if needed
            'args': args,
            'loss': loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}, change point is {change_points_auto}")

    return acc


def ce_loss(input_logits, target_logits):
    target_dist = torch.softmax(target_logits, dim=1)
    input_dist = F.log_softmax(input_logits, dim=1)
    loss = F.kl_div(input_dist, target_dist, reduction='batchmean')
    return loss


def main():
    parser = argparse.ArgumentParser(description="Training script for your model.")
    # Define arguments (these should match the YAML keys)
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_pertub', type=int, default=1)
    parser.add_argument('--zo_lr', type=float, default=1e-6)
    parser.add_argument('--zo_eps_cge', type=float, default=1e-5)
    parser.add_argument('--zo_eps_rge', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--change_point', type=int, default=500)
    parser.add_argument('--prune_interval', type=int, default=200)
    parser.add_argument('--prune_ratio', type=float, default=0.1)
    parser.add_argument('--prompt_trainer', type=str, default='zo-pruned', choices=['cma-es', 'zo', 'fo', 'zo-pruned'])
    parser.add_argument('--smoothness_estimator', type=str, default='zo', choices=['zo', 'fo', 'none'])
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--backbone', type=str,
                        default='ViT-B/16')  # choose from 'RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'
    parser.add_argument('--root_path', type=str, default='<Your Root Path>')
    parser.add_argument('--shots', type=int, default=16)
    parser.add_argument('--sigma', type=float, default=0.4)

    args = parser.parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    cfg['subsample_classes'] = 'all'
    # merge cfg to args
    # kwargs = vars(args)
    for key, value in cfg.items():
        if not hasattr(args, key):
            parser.add_argument('--' + key, default=value)
    args = parser.parse_args()

    args_dict = vars(args)

    wandb_run_name = str(cfg["dataset"]) + '-rho-' + str(args.rho) + '-std-' + str(
        args.std) + '-sigma-' + str(args.sigma) + '-zo-eps-cge-' + str(args.zo_eps_cge)
    wandb.init(name=wandb_run_name, config=args_dict)
    # load CLIP
    global preprocess
    clip_model, preprocess = clip.load(args.backbone)
    clip_model.eval()

    # ---------------------------------------- run ------------------------------------
    results = {"1": [], "2": [], "3": []}
    if cfg["dataset"] == 'imagenet':
        if args.shots in [8, 16]:
            cfg["budget"] = 4000
        elif args.shots in [4, 2]:
            cfg["budget"] = 4000
        else:
            cfg["budget"] = 2000
    else:
        if args.shots in [8, 16]:
            cfg["budget"] = 8000
        elif args.shots in [4, 2]:
            cfg["budget"] = 4000
        else:
            cfg["budget"] = 2000

    cfg["shots"] = args.shots
    for seed in [1]:
        cfg['seed'] = seed
        print("\nRunning configs.")
        print(cfg, "\n")

        random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        print("Preparing dataset.")

        if cfg["dataset"] == 'imagenet':
            dataset = ImageNet(cfg, os.path.join(args.root_path, 'datasets'), cfg['shots'], preprocess)
            test_loader = torch.utils.data.DataLoader(dataset.test, batch_size=32, num_workers=8, shuffle=False)
            val_loader = torch.utils.data.DataLoader(dataset.test, batch_size=32, num_workers=8, shuffle=False)
            train_loader_F = torch.utils.data.DataLoader(dataset.train, batch_size=256, num_workers=8, shuffle=True)
        else:
            dataset = build_dataset(cfg, cfg['dataset'], os.path.join(args.root_path, 'datasets'), cfg['shots'])
            val_loader = build_data_loader(data_source=dataset.val, batch_size=64, is_train=False, tfm=preprocess,
                                           shuffle=False)
            test_loader = build_data_loader(data_source=dataset.test, batch_size=64, is_train=False, tfm=preprocess,
                                            shuffle=False)
            train_loader_F = build_data_loader(data_source=dataset.train_x, batch_size=args.batch_size,
                                               tfm=train_transform, is_train=True, shuffle=True)

        acc = run(args, cfg, dataset, clip_model, val_loader, test_loader, train_loader_F, run_name=wandb_run_name)

        results[str(seed)].append(acc)
    print("Dataset: %s" % (cfg["dataset"]))
    print("Backbone: %s" % (args.backbone))
    print("Results on shots: [1, 2, 4, 8, 16]")
    for seed in ["1"]:
        print("Results on seed %s: %s" % (seed, results[seed]))


if __name__ == '__main__':
    main()
