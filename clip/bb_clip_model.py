
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import torchvision.transforms as transforms

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

# from utils import *

def ce_loss(input_logits, target_logits):
    target_dist = torch.softmax(target_logits, dim=1)
    input_dist = F.log_softmax(input_logits, dim=1)
    loss = F.kl_div(input_dist, target_dist, reduction='batchmean')
    return loss

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

class BBCLIP(nn.Module):
    def __init__(self, args, cfg, classnames, clip_model):
        super(BBCLIP, self).__init__()
        # Initialize the main component of the model

        self.adapter = nn.Sequential(
                nn.Linear(len(classnames), 256),
                nn.ReLU(),
                nn.Linear(256, len(classnames))
                ).cuda()
        self.clip_model = clip_model
        self.enable_adapter = args.enable_adapter

        # Initialize the prompt learner parameter

        self.device = torch.cuda.current_device()
        # self.train_features, self.train_labels = pre_load_features(cfg, "train", clip_model, train_loader_F)
        n_cls = len(classnames)
        n_ctx = 4
        self.args = args
        ctx_init = 'a photo of a'
        self.dtype = self.clip_model.dtype
        ctx_dim = self.clip_model.ln_final.weight.shape[0]

        self.n_cls = n_cls
        self.n_ctx = n_ctx

        # use given words to initialize context vectors
        ctx_init = ctx_init.replace("_", " ")
        n_ctx = len(ctx_init.split(" "))
        prompt = clip.tokenize(ctx_init).cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(prompt).type(self.dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :].cuda()
        prompt_prefix = ctx_init
        self.n_ctx = n_ctx

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = ctx_vectors
        self.bias = torch.rand_like(ctx_vectors)
        self.prompt_prefix = prompt_prefix
        self.get_prefix_suffix_token(classnames, self.clip_model)

        self.linear = nn.Linear(cfg["intrinsic_dim"], n_ctx * ctx_vectors.shape[1], bias=False)

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

    def forward(self, input=None, prompt_embedding=None, train_features=None, target=None):
        # obtain the encoded text_features
        # assert input is not None and train_features is not None
        ctx = self.ctx  # p_0
        if prompt_embedding is not None:
            ctx = ctx + self.linear(prompt_embedding).reshape(self.n_ctx, -1)  # p_0 + Az
        else:
            ctx = ctx + self.bias

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

        text_features = self.text_encoder(prompts.float(), self.tokenized_prompts.float())
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        if train_features is not None:
            image_features = train_features
        else:
            image_features = self.clip_model.encode_image(input)
        logits = 100. * image_features.float() @ text_features.T.float()

        if self.args.enable_adapter:
            update_logits = logits + self.adapter(logits)
            loss = F.cross_entropy(update_logits, target.long())
            loss += self.cfg["weight"] * ce_loss(logits, update_logits)
        else:
            update_logits = logits
            loss = F.cross_entropy(logits, target.long())

        return loss, update_logits

    def update_context_prompt(self, es):
        solutions = es.ask()  # list of numpy array. [numpy.ndarray]. len(solutions) = cfg["popsize"]
        inputs = torch.tensor(np.array(solutions)).cuda().float()
        self.bias = self.linear(inputs).reshape(-1, self.n_ctx, self.ctx.shape[1]).mean(dim=0)
