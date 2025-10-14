import json
import tokenizers
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F
from model_ssacl import ssacl
from tqdm import tqdm
from PIL import Image
import numpy as np
import os


def evaluate_retrieval(similarity_matrix, ground_truth=None, k_list=[1, 5, 10, 100]):
    """
    Evaluate Recall@K and Precision@K for image-text or text-image retrieval.

    Args:
        similarity_matrix (np.ndarray): shape [N, N], entry (i, j) is the similarity between image i and text j.
        ground_truth (np.ndarray or None): if None, assumes image i matches text i.
        k_list (List[int]): list of K values to evaluate, e.g., [1, 5, 10].

    Returns:
        dict: mapping each K to R@K and P@K (in percentage, rounded to 2 decimals).
    """
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Expected a square matrix: numbers of images and texts must match."
    num_queries = similarity_matrix.shape[0]
    if ground_truth is None:
        ground_truth = np.arange(num_queries)

    results = {}
    for k in k_list:
        correct_at_k = 0
        precision_total = 0

        for i in range(num_queries):
            # sort by similarity in descending order
            retrieved_indices = np.argsort(-similarity_matrix[i])[:k]
            if ground_truth[i] in retrieved_indices:
                correct_at_k += 1
                precision_total += 1.0 / k  # single-hit precision contribution

        recall_at_k = correct_at_k / num_queries
        precision_at_k = precision_total / num_queries

        results[f"R@{k}"] = round(recall_at_k * 100, 2)
        results[f"P@{k}"] = round(precision_at_k * 100, 2)

    return results


def i2t(sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (N, max_n_word, d) matrix of captions
    CapLens: (N) array of caption lengths
    sims: (N, N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5))
    top10 = np.zeros((npts, 10))
    top100 = np.zeros((npts, 100))

    for index in range(npts):
        inds = np.argsort(-1 * sims[index])
        tmp = np.where(inds == index)[0][0]
        ranks[index] = tmp
        top1[index] = inds[0]
        top5[index] = inds[0:5]
        top10[index] = inds[0:10]
        top100[index] = inds[0:100]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    MRR = np.sum(1 / (ranks + 1)) / len(ranks)
    if return_ranks:
        return (r1, r5, r10, medr, meanr, MRR), (ranks, top1, top5, top10, top100)
    else:
        return (r1, r5, r10, medr, meanr, MRR)


def calcu_tfpn(preds_gen_reports_np, preds_ref_reports_np):
    # Boolean arrays of shape (num_classes x num_reports)
    tp = np.logical_and(preds_gen_reports_np, preds_ref_reports_np)
    fp = np.logical_and(preds_gen_reports_np, ~preds_ref_reports_np)
    fn = np.logical_and(~preds_gen_reports_np, preds_ref_reports_np)
    tn = np.logical_and(~preds_gen_reports_np, ~preds_ref_reports_np)

    # Sum TP/FP/FN/TN per report (per column)
    tp_example = tp.sum(axis=0)
    fp_example = fp.sum(axis=0)
    fn_example = fn.sum(axis=0)
    tn_example = tn.sum(axis=0)

    precision_example = tp_example / (tp_example + fp_example)
    recall_example = tp_example / (tp_example + fn_example)
    f1_example = (2 * tp_example) / (2 * tp_example + fp_example + fn_example)
    acc_example = (tp_example + tn_example) / (tp_example + tn_example + fp_example + fn_example)
    spec_example = tn_example / (tn_example + fp_example)
    npv_example = tn_example / (tn_example + fn_example)

    # Handle zero-division
    precision_example[np.isnan(precision_example)] = 0.0
    recall_example[np.isnan(recall_example)] = 0.0
    f1_example[np.isnan(f1_example)] = 0.0
    acc_example[np.isnan(acc_example)] = 0.0
    spec_example[np.isnan(spec_example)] = 0.0
    npv_example[np.isnan(npv_example)] = 0.0

    # Mean across reports
    precision_example = float(precision_example.mean())
    recall_example = float(recall_example.mean())
    f1_example = float(f1_example.mean())
    acc_example = float(acc_example.mean())
    spec_example = float(spec_example.mean())
    npv_example = float(npv_example.mean())

    print('acc: ', acc_example)
    print('sens./recall: ', recall_example)
    print('spec.: ', spec_example)
    print('ppv/precision: ', precision_example)
    print('npv: ', npv_example)
    print('f1: ', f1_example)


def load_model(pretrained_path) -> torch.nn.Module:
    """ Load the model.
    Just load the model and leave the post-processing in self.post_process()
    Input:
        - Necessary arguments to load the pretrained model. (e.g. ckpt_dir)
    Return:
        - model(torch.nn.Module): Any torch model with loaded parameters
    """
    # step-1: init model
    model = ssacl(norm_pix_loss=False, T=0.03, lam=0.9, SR=0.0)

    # step-2: load pretrain parameters
    checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
    try:
        model_dict = checkpoint['state_dict']
    except:
        model_dict = checkpoint['model']
    msg = model.load_state_dict(model_dict, strict=False)
    print(msg)

    model.train(False)
    return model


class CustomDataset(Dataset):
    def __init__(self, data_pth, transform=None):
        self.transform = transform
        with open(data_pth, 'r') as f:
            self.ann = json.load(f)
        self.tokenizer = tokenizers.Tokenizer.from_file("data/mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.tokenizer.enable_truncation(max_length=100)
        self.tokenizer.enable_padding(length=100)

        encoded = [self.tokenizer.encode(a['caption']) for a in self.ann]
        self.report_ids = torch.stack([torch.tensor(e.ids) for e in encoded])
        self.attention_mask = torch.stack([torch.tensor(e.attention_mask) for e in encoded])
        self.type_ids = torch.stack([torch.tensor(e.type_ids) for e in encoded])

        self.img_root = '/path/to/chexpert/'

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_root, self.ann[idx]['image'].replace('CheXpert-v1.0', 'CheXpert-v1.0-small'))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        pathology_labels = (torch.abs(torch.nan_to_num(torch.tensor(self.ann[idx]['labels']), nan=0))).long()
        return image, torch.tensor([idx], dtype=torch.long), pathology_labels


preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4978], std=[0.2449])
])


def run():
    data_pth = 'data/chexpert_5x200_test.json'

    model = load_model('pretrained_weights/ssacl-best.pth')
    model = model.cuda()

    batch_size = 100
    dataset = CustomDataset(data_pth, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ids = dataset.report_ids.cuda()
    attention_mask = dataset.attention_mask.cuda()
    type_ids = dataset.type_ids.cuda()
    latent_report = model.bert_encoder(None, ids, None, attention_mask, type_ids)
    latent_report_global = latent_report.logits
    latent_report_global = latent_report_global / latent_report_global.norm(dim=-1, keepdim=True)

    preds = []
    gts = []
    plabels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            images, labels, plabel = batch
            images = images.cuda()

            latent = model.forward_img_encoder_nomask(images)
            latent_img = model.img_mlp(latent)
            latent_img = latent_img[:, 1:, :]
            latent_img_global = latent_img.mean(dim=1)

            latent_img_global = latent_img_global / latent_img_global.norm(dim=-1, keepdim=True)

            sim = latent_img_global @ latent_report_global.T
            sim = F.softmax(sim, -1)

            preds.append(sim.cpu().numpy())
            gts.append(labels.squeeze())
            plabels.append(plabel)

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    plabels = np.concatenate(plabels)

    preds = preds.T
    gts = gts.T

    ri2t, ranktop = i2t(preds, return_ranks=True)
    ranks, top1, top5, top10, top100 = ranktop

    top1_label = [plabels[int(i)] for i in top1]
    top5_label = [sum((plabels[int(i)] == plabels[g]) * plabels[g]) / sum(plabels[int(i)]) for g, t5 in enumerate(top5) for i in t5]
    top10_label = [sum((plabels[int(i)] == plabels[g]) * plabels[g]) / sum(plabels[int(i)]) for g, t10 in enumerate(top10) for i in t10]
    top100_label = [sum((plabels[int(i)] == plabels[g]) * plabels[g]) / sum(plabels[int(i)]) for g, t100 in enumerate(top100) for i in t100]

    p5 = 100. * sum(top5_label) / len(top5_label)
    p10 = 100. * sum(top10_label) / len(top10_label)
    p100 = 100. * sum(top100_label) / len(top100_label)

    Pi2t = {'p@5': p5, 'p@10': p10, 'p@100': p100}
    Ri2t = {'r@1': ri2t[0], 'r@5': ri2t[1], 'r@10': ri2t[2], 'mrr': ri2t[-1]}

    for k, v in Ri2t.items():
        print(f'{k}: {v}')
    for k, v in Pi2t.items():
        print(f'{k}: {v}')


if __name__ == "__main__":
    run()
