from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def pre_load_features(cfg, split, clip_model, loader, norm=True):
    features, labels = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model.encode_image(images)
            if norm:
                image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
            labels.append(target)

    features, labels = torch.cat(features), torch.cat(labels)  
    return features, labels

def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

    return cache_keys, cache_values


def compute_eps(args, bb_clip, prompt_learner, zo_trainer):
    """
    Compute the eps for the SAM
    :return: eps
    """
    if args.smoothness_estimator != 'none':
        # compute the analytic solution for the perturbs
        if args.smoothness_estimator == 'fo':
            raise NotImplementedError
            # # print(f'Use FO Smoothness Estimator')
            # # query gradients with BP
            # prompts, tokenized_prompts = prompt_learner()
            # loss, _ = bb_clip(
            #     prompts=prompts,
            #     tokenized_prompts=tokenized_prompts,
            #     train_features=prompt_learner.train_features,
            #     train_labels=prompt_learner.train_labels,
            # )
            # loss.backward()
            # param_grad = prompt_learner.input_param.grad
        elif args.smoothness_estimator == 'zo':
            # print(f'Use ZO Smoothness Estimator')
            # query grad with RGE or CGE
            zo_trainer.zo_step_cge(
                train_features=prompt_learner.train_features,
                train_labels=prompt_learner.train_labels,
            )
            param_grad = list(zo_trainer.grad_tmp.items())[0][1]

        # Compute the epsilon
        grad_norm = param_grad.norm(p=2)
        eps = args.rho * param_grad / (grad_norm + 1e-12)
    else:
        eps = 0
    return eps

def evaluate_model(args, bb_clip, prompt_learner, test_features, test_labels):
    """

    :return:
    """
    with torch.no_grad():
        prompts, tokenized_prompts = prompt_learner()
        _, test_logits = bb_clip(
            prompts=prompts,
            tokenized_prompts=tokenized_prompts,
            train_features=test_features,
            train_labels=test_labels,
        )
        acc = cls_acc(test_logits, test_labels)
    return acc