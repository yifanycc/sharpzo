import torch
import torch.nn.functional as F
import pdb
import time
from utils import *

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

def _model_forward(prompts, tokenized_prompts, text_encoder, train_features, train_labels):

    text_features = text_encoder(prompts.float(), tokenized_prompts.float())
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    logits = 100. * train_features.float() @ text_features.T.float()

    loss = F.cross_entropy(logits, train_labels.long())
    return loss


# def measure_smoothness(metrics, args, cfg, prompts, tokenized_prompts, clip_model, train_loader, epsilon=1e-4, num_samples=1, device='cuda'):
#     optimizer = torch.optim.Adam(clip_model.parameters(), lr=1e-3)
#     for name, param in clip_model.named_parameters():
#         param.requires_grad = True
#
#     train_features, train_labels = pre_load_features(cfg, "train", clip_model, train_loader)
#     text_encoder = TextEncoder(clip_model).float().cuda()
#
#     if metrics == 'definition':
#         # model.train()  # Set model to evaluation mode
#         L_values = []  # To store local smoothness values
#
#             # if i >= num_samples:
#             #     break  # Only sample `num_samples` batches
#         time1 = time.time()
#         # for name, param in model.named_parameters():
#         #     param.requires_grad = True
#         loss = _model_forward(prompts, tokenized_prompts, text_encoder, train_features, train_labels)
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         original_grad = []
#         time1 = time.time()
#         # pdb.set_trace()
#         for name, param in clip_model.named_parameters():
#             if param.grad is not None:
#                 original_grad.append(param.grad.clone().cpu())
#
#         time2 = time.time()
#         perturbation = []
#         seed = 42  # Set your desired seed value
#         torch.manual_seed(seed)
#         for name, param in clip_model.named_parameters():
#             z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
#                              dtype=param.data.dtype)
#             param.data = param.data + 1 * z * epsilon
#             perturbation.append(1 * z * epsilon)
#
#         loss = _model_forward(prompts, tokenized_prompts, text_encoder, train_features, train_labels)
#         optimizer.zero_grad()
#         loss.backward(retain_graph=True)
#         time3 = time.time()
#         perturbed_grad = []
#         for name, param in clip_model.named_parameters():
#             if param.grad is not None:
#                 perturbed_grad.append(param.grad.clone().cpu())
#
#         for param, delta in zip(clip_model.parameters(), perturbation):
#             param.data.sub_(delta)  # Undo perturbation
#
#         grad_diff = torch.norm(torch.cat([g.view(-1) for g in perturbed_grad]) -
#                                torch.cat([g.view(-1) for g in original_grad]))
#         input_diff = torch.norm(torch.cat([p.view(-1) for p in perturbation]))
#         L_local = grad_diff / input_diff
#         # pdb.set_trace()
#         return L_local.item()
#     elif metrics == 'hessian':
#         model.eval()  # Set model to evaluation mode
#         L_values = []  # Store smoothness values
#
#         for i, (image, target) in enumerate(train_loader):
#             if i >= num_samples:
#                 break  # Only sample `num_samples` batches
#
#             image, target = image.to(device), target.to(device)
#
#             # Forward pass to compute logits
#             with torch.no_grad():
#                 image_features = clip_model.encode_image(image)  # CLIP image encoding
#                 text_features = prompt_learner()  # Prompt learner features
#                 logits = 100. * image_features.float() @ text_features.T.float()
#
#             logits.requires_grad_(True)
#
#             # Define the loss function
#             def loss_fn(logits):
#                 update_logits = logits + model(logits)
#                 return F.cross_entropy(update_logits, target.long())
#
#             # Compute the Hessian matrix of the loss function with respect to logits
#             H = hessian(loss_fn, logits)
#
#             # Compute the spectral norm (maximum eigenvalue of the Hessian)
#             # Use power iteration to approximate the largest eigenvalue
#             def power_iteration(H, num_iters=10):
#                 v = torch.randn(H.shape[0], device=H.device)  # Random vector
#                 for _ in range(num_iters):
#                     v = torch.mv(H, v)  # Matrix-vector multiplication
#                     v = v / torch.norm(v)  # Normalize the vector
#                 eigenvalue = torch.dot(v, torch.mv(H, v))  # Rayleigh quotient
#                 return eigenvalue
#
#             L_local = power_iteration(H)
#             L_values.append(L_local.item())
#
#         # Average the smoothness values over sampled batches
#         L_smoothness = sum(L_values) / len(L_values)
#         return L_smoothness
#     else:
#             raise NotImplementedError