import csv
import os
import re
import tempfile

import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, f1_score, recall_score, precision_score
import torch
import torch.nn as nn
import pandas as pd

from CheXbert.src.constants import CONDITIONS
from CheXbert.src.label import label
from CheXbert.src.models.bert_labeler import bert_labeler
from pycocoevalcap.cider.cider import Cider

from typing import List, Dict


path_chexbert_weights="pretrained_weights/chexbert.pth"


def compute_NLG_scores(nlg_metrics: List[str], gen_sents_or_reports: List[str], ref_sents_or_reports: List[str]) -> Dict[str, float]:
    def convert_for_pycoco_scorer(sents_or_reports: List[str]):
        """
        The compute_score methods of the scorer objects require the input not to be list[str],
        but of the form:
        generated_reports =
        {
            "image_id_0" = ["1st generated report"],
            "image_id_1" = ["2nd generated report"],
            ...
        }

        Hence we convert the generated/reference sentences/reports into the appropriate format and also tokenize them
        following Nicolson's (https://arxiv.org/pdf/2201.09405.pdf) implementation (https://github.com/aehrc/cvt2distilgpt2/blob/main/transmodal/metrics/chen.py):
        see lines 132 and 133
        """
        sents_or_reports_converted = {}
        for num, text in enumerate(sents_or_reports):
            sents_or_reports_converted[str(num)] = [re.sub(' +', ' ', text.replace(".", " ."))]

        return sents_or_reports_converted
    """
    Computes NLG metrics that are specified in metrics list (1st input argument):
        - Bleu 1-4
        - Meteor
        - Rouge-L
        - Cider-D

    Returns a dict that maps from the metrics specified to the corresponding scores.
    """
    scorers = {}
    if "bleu" in nlg_metrics:
        scorers["bleu"] = Bleu(4)
    if "meteor" in nlg_metrics:
        scorers["meteor"] = Meteor()
    if "rouge" in nlg_metrics:
        scorers["rouge"] = Rouge()  # this is actually the Rouge-L score, even if the class name only says Rouge
    if "cider" in nlg_metrics:
        scorers["cider"] = Cider()  # this is actually the Cider-D score, even if the class name only says Cider

    gen_sents_or_reports = convert_for_pycoco_scorer(gen_sents_or_reports)
    ref_sents_or_reports = convert_for_pycoco_scorer(ref_sents_or_reports)

    nlg_scores = {}

    for metric_name, scorer in scorers.items():
        score, _ = scorer.compute_score(ref_sents_or_reports, gen_sents_or_reports)
        if metric_name == "bleu":
            nlg_scores["bleu_1"] = score[0]
            nlg_scores["bleu_2"] = score[1]
            nlg_scores["bleu_3"] = score[2]
            nlg_scores["bleu_4"] = score[3]
        else:
            nlg_scores[metric_name] = score

    return nlg_scores

def compute_clinical_efficacy_scores(language_model_scores: dict, gen_reports: List[str], ref_reports: List[str]):
    """
    This function computes:
        - micro average CE scores over all 14 conditions
        - micro average CE scores over 5 conditions ("Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion")
        -> this is done following Miura (https://arxiv.org/pdf/2010.10042.pdf)
        - (micro) average CE scores of each condition
        - example-based CE scores over all 14 conditions
        -> this is done following Nicolson (https://arxiv.org/pdf/2201.09405.pdf)

    To compute these scores, we first need to get the disease labels extracted by CheXbert for both the generated and reference reports.
    This is done by the (nested) function "get_chexbert_labels_for_gen_and_ref_reports". Inside this function, there is another function
    called "label" from the module src/CheXbert/src/label.py that extracts these labels requiring 2 input arguments:
        1. chexbert (nn.Module): instantiated chexbert model
        2. csv_path (str): path to the csv file with the reports. The csv file has to have 1 column titled "Report Impression"
        under which the reports can be found

    We use a temporary directory to create the csv files for the generated and reference reports.

    The function label returns preds_gen_reports and preds_ref_reports respectively, which are List[List[int]],
    with the outer list always having len=14 (for 14 conditions, specified in CheXbert/src/constants.py),
    and the inner list has len=num_reports.

    E.g. the 1st inner list could be [2, 1, 0, 3], which means the 1st report has label 2 for the 1st condition (which is 'Enlarged Cardiomediastinum'),
    the 2nd report has label 1 for the 1st condition, the 3rd report has label 0 for the 1st condition, the 4th and final report label 3 for the 1st condition.

    There are 4 possible labels:
        0: blank/NaN (i.e. no prediction could be made about a condition, because it was no mentioned in a report)
        1: positive (condition was mentioned as present in a report)
        2: negative (condition was mentioned as not present in a report)
        3: uncertain (condition was mentioned as possibly present in a report)

    To compute the micro average scores (i.e. all the scores except of the example-based scores), we follow the implementation of the paper
    by Miura et. al., who considered the negative and blank/NaN to be one whole negative class, and positive and uncertain to be one whole positive class.
    For reference, see lines 141 and 143 of Miura's implementation: https://github.com/ysmiura/ifcc/blob/master/eval_prf.py#L141,
    where label 3 is converted to label 1, and label 2 is converted to label 0.

    To compute the example-based scores, we follow Nicolson's implementation, who considered blank/NaN, negative and uncertain to be the negative class,
    and only positive to be the positive class. Meaning labels 2 and 3 are converted to label 0.
    """

    def get_chexbert():
        model = bert_labeler()
        model = nn.DataParallel(model)  # needed since weights were saved with nn.DataParallel
        checkpoint = torch.load(path_chexbert_weights, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.cuda()
        model.eval()

        return model

    def get_chexbert_labels_for_gen_and_ref_reports():
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_gen_reports_file_path = os.path.join(temp_dir, "gen_reports.csv")
            csv_ref_reports_file_path = os.path.join(temp_dir, "ref_reports.csv")

            header = ["Report Impression"]

            with open(csv_gen_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[gen_report] for gen_report in gen_reports])

            with open(csv_ref_reports_file_path, "w") as fp:
                csv_writer = csv.writer(fp)
                csv_writer.writerow(header)
                csv_writer.writerows([[ref_report] for ref_report in ref_reports])

            # preds_*_reports are List[List[int]] with the labels extracted by CheXbert (see doc string for details)
            preds_gen_reports = label(chexbert, csv_gen_reports_file_path)
            preds_ref_reports = label(chexbert, csv_ref_reports_file_path)

        return preds_gen_reports, preds_ref_reports

    def compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports):
        def convert_labels_like_miura(preds_reports: List[List[int]]):
            """
            See doc string of update_clinical_efficacy_scores function for more details.
            Miura (https://arxiv.org/pdf/2010.10042.pdf) considers blank/NaN (label 0) and negative (label 2) to be the negative class,
            and positive (label 1) and uncertain (label 3) to be the positive class.

            Thus we convert label 2 -> label 0 and label 3 -> label 1.
            """
            def convert_label(label: int):
                if label == 2:
                    return 0
                elif label == 3:
                    return 0
                else:
                    return label

            preds_reports_converted = [[convert_label(label) for label in condition_list] for condition_list in preds_reports]

            return preds_reports_converted

        preds_gen_reports_converted = convert_labels_like_miura(preds_gen_reports)
        preds_ref_reports_converted = convert_labels_like_miura(preds_ref_reports)

        # for the CE scores, we follow Miura (https://arxiv.org/pdf/2010.10042.pdf) in micro averaging them over these 5 conditions:
        five_conditions_to_evaluate = {"Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"}

        total_preds_gen_reports_5_conditions = []
        total_preds_ref_reports_5_conditions = []

        # we also compute the micro average over all 14 conditions:
        total_preds_gen_reports_14_conditions = []
        total_preds_ref_reports_14_conditions = []

        preds_gen_reports_5_conditions = []
        preds_ref_reports_5_conditions = []

        # iterate over the 14 conditions
        for preds_gen_reports_condition, preds_ref_reports_condition, condition in zip(preds_gen_reports_converted, preds_ref_reports_converted, CONDITIONS):
            if condition in five_conditions_to_evaluate:
                total_preds_gen_reports_5_conditions.extend(preds_gen_reports_condition)
                total_preds_ref_reports_5_conditions.extend(preds_ref_reports_condition)

                preds_gen_reports_5_conditions.append(preds_gen_reports_condition)
                preds_ref_reports_5_conditions.append(preds_ref_reports_condition)

            total_preds_gen_reports_14_conditions.extend(preds_gen_reports_condition)
            total_preds_ref_reports_14_conditions.extend(preds_ref_reports_condition)

            # compute and save scores for the given condition
            precision, recall, f1, _ = precision_recall_fscore_support(preds_ref_reports_condition, preds_gen_reports_condition, average="binary")
            acc = accuracy_score(preds_ref_reports_condition, preds_gen_reports_condition)

            language_model_scores["report"]["CE"][condition]["precision"] = precision
            language_model_scores["report"]["CE"][condition]["recall"] = recall
            language_model_scores["report"]["CE"][condition]["f1"] = f1
            language_model_scores["report"]["CE"][condition]["acc"] = acc

        # compute and save scores for all 14 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_14_conditions, total_preds_gen_reports_14_conditions)

        cmn_metrics = compute_mlc(np.array(preds_ref_reports_converted).transpose(), np.array(preds_gen_reports_converted).transpose(), CONDITIONS)

        language_model_scores["report"]["CE"]["precision_micro_all"] = precision
        language_model_scores["report"]["CE"]["recall_micro_all"] = recall
        language_model_scores["report"]["CE"]["f1_micro_all"] = f1
        language_model_scores["report"]["CE"]["acc_all"] = acc
        for k, v in cmn_metrics.items():
            language_model_scores["main_metrics"][f'CE-14_{k}'] = v

        # compute and save scores for the 5 conditions
        precision, recall, f1, _ = precision_recall_fscore_support(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions, average="binary")
        acc = accuracy_score(total_preds_ref_reports_5_conditions, total_preds_gen_reports_5_conditions)

        cmn_metrics = compute_mlc(np.array(preds_ref_reports_5_conditions).transpose(), np.array(preds_gen_reports_5_conditions).transpose(), CONDITIONS)

        language_model_scores["report"]["CE"]["precision_micro_5"] = precision
        language_model_scores["report"]["CE"]["recall_micro_5"] = recall
        language_model_scores["report"]["CE"]["f1_micro_5"] = f1
        language_model_scores["report"]["CE"]["acc_5"] = acc
        for k, v in cmn_metrics.items():
            language_model_scores["main_metrics"][f'CE-5_{k}'] = v

    def compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports):
        """
        example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        """
        preds_gen_reports_np = np.array(preds_gen_reports)  # array of shape (14 x num_reports), 14 for 14 conditions
        preds_ref_reports_np = np.array(preds_ref_reports)  # array of shape (14 x num_reports)

        # convert label 1 to True and everything else (i.e. labels 0, 2, 3) to False
        # (effectively doing the label conversion as done by Nicolson, see doc string of compute_clinical_efficacy_scores for more details)
        preds_gen_reports_np = preds_gen_reports_np == 1
        preds_ref_reports_np = preds_ref_reports_np == 1

        tp = np.logical_and(preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fp = np.logical_and(preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        fn = np.logical_and(~preds_gen_reports_np, preds_ref_reports_np)  # bool array of shape (14 x num_reports)
        tn = np.logical_and(~preds_gen_reports_np, ~preds_ref_reports_np)  # bool array of shape (14 x num_reports)

        # sum up the TP, FP, FN and TN for each report (i.e. for each column)
        tp_example = tp.sum(axis=0)  # int array of shape (num_reports)
        fp_example = fp.sum(axis=0)  # int array of shape (num_reports)
        fn_example = fn.sum(axis=0)  # int array of shape (num_reports)
        tn_example = tn.sum(axis=0)  # int array of shape (num_reports)

        # compute the scores for each report
        precision_example = tp_example / (tp_example + fp_example)  # float array of shape (num_reports)
        recall_example = tp_example / (tp_example + fn_example)  # float array of shape (num_reports)
        f1_example = (2 * tp_example) / (2 * tp_example + fp_example + fn_example)  # float array of shape (num_reports)
        acc_example = (tp_example + tn_example) / (tp_example + tn_example + fp_example + fn_example)  # float array of shape (num_reports)

        # since there can be cases of zero division, we have to replace the resulting nan values with 0.0
        precision_example[np.isnan(precision_example)] = 0.0
        recall_example[np.isnan(recall_example)] = 0.0
        f1_example[np.isnan(f1_example)] = 0.0
        acc_example[np.isnan(acc_example)] = 0.0

        # finally, take the mean over the scores for all reports
        precision_example = float(precision_example.mean())
        recall_example = float(recall_example.mean())
        f1_example = float(f1_example.mean())
        acc_example = float(acc_example.mean())

        language_model_scores["report"]["CE"]["precision_example_all"] = precision_example
        language_model_scores["report"]["CE"]["recall_example_all"] = recall_example
        language_model_scores["report"]["CE"]["f1_example_all"] = f1_example
        language_model_scores["report"]["CE"]["acc_example_all"] = acc_example

    chexbert = get_chexbert()
    preds_gen_reports, preds_ref_reports = get_chexbert_labels_for_gen_and_ref_reports()

    compute_micro_average_CE_scores(preds_gen_reports, preds_ref_reports)
    compute_example_based_CE_scores(preds_gen_reports, preds_ref_reports)

def compute_language_model_scores(gen_and_ref_reports):

    def compute_report_level_scores():
        gen_reports = gen_and_ref_reports["generated_reports"]
        ref_reports = gen_and_ref_reports["reference_reports"]

        nlg_metrics = ["bleu", "meteor", "rouge", "cider"]
        nlg_scores = compute_NLG_scores(nlg_metrics, gen_reports, ref_reports)

        for nlg_metric_name, score in nlg_scores.items():
            language_model_scores["report"][nlg_metric_name] = score
            language_model_scores["main_metrics"][nlg_metric_name] = score

        compute_clinical_efficacy_scores(language_model_scores, gen_reports, ref_reports)

    def create_language_model_scores_dict():
        language_model_scores = {}

        # on report-level, we evalute on:
        # BLEU 1-4
        # METEOR
        # ROUGE-L
        # Cider-D
        # CE scores (P, R, F1, acc)
        language_model_scores["report"] = {f"bleu_{i}": None for i in range(1, 5)}
        language_model_scores["report"]["meteor"] = None
        language_model_scores["report"]["rouge"] = None
        language_model_scores["report"]["cider"] = None
        language_model_scores["report"]["CE"] = {
            # following Miura (https://arxiv.org/pdf/2010.10042.pdf), we evaluate the micro average CE scores over these 5 diseases/conditions:
            # "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"
            "precision_micro_5": None,
            "recall_micro_5": None,
            "f1_micro_5": None,
            "acc_5": None,

            # we additionally compute the micro average CE scores over all conditions
            "precision_micro_all": None,
            "recall_micro_all": None,
            "acc_all": None
        }

        # we also compute the CE scores for each of the 14 conditions individually
        for condition in CONDITIONS:
            language_model_scores["report"]["CE"][condition] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "acc": None
            }

        # following Nicolson (https://arxiv.org/pdf/2201.09405.pdf), we evaluate the example-based CE scores over all conditions
        # example-based means precision/recall/F1/acc are computed for each report, and then these scores are averaged over all reports
        language_model_scores["report"]["CE"]["precision_example_all"] = None
        language_model_scores["report"]["CE"]["recall_example_all"] = None
        language_model_scores["report"]["CE"]["f1_example_all"] = None
        language_model_scores["report"]["CE"]["acc_example_all"] = None

        # on sentence-level, we only evaluate on METEOR, since this metric gives meaningful scores on sentence-level (as opposed to e.g. BLEU)
        # we distinguish between generated sentences for all, normal, and abnormal regions
        for subset in ["all", "normal", "abnormal"]:
            language_model_scores[subset] = {"meteor": None}

        # we also compute these scores for each region individually
        language_model_scores["region"] = {}
        for region_name in ANATOMICAL_REGIONS:
            language_model_scores["region"][region_name] = {"meteor": None}

        # and finally, on sentence-level we also compute the ratio of the meteor scores for when:
        #   - a generated sentence is paired with its corresponding reference sentence of a given image (value for the numerator)
        #   vs
        #   - a generated sentence is paired with all other non-corresponding reference sentences of a given image (value for the denominator)
        #
        # the numerator value is already computed by language_model_scores["all"]["meteor"], since this is exactly the meteor score for when the generated sentences
        # are paired with their corresponding reference sentences. Hence only the denominator value has to be calculated separately
        language_model_scores["all"]["meteor_ratio"] = None

        language_model_scores["main_metrics"] = {}

        return language_model_scores

    language_model_scores = create_language_model_scores_dict()

    compute_report_level_scores()

    return language_model_scores


def compute_mlc(gt, pred, label_set):
    res_mlc = {}
    res_mlc['F1_MACRO'] = f1_score(gt, pred, average="macro")
    res_mlc['F1_MICRO'] = f1_score(gt, pred, average="micro")
    res_mlc['RECALL_MACRO'] = recall_score(gt, pred, average="macro")
    res_mlc['RECALL_MICRO'] = recall_score(gt, pred, average="micro")
    res_mlc['PRECISION_MACRO'] = precision_score(gt, pred, average="macro")
    res_mlc['PRECISION_MICRO'] = precision_score(gt, pred, average="micro")

    return res_mlc

def sample_f1_scores(y_true, y_pred):
    f1_scores = []
    for i in range(len(y_true)):  # 遍历每个样本
        p = precision_score(y_true[i], y_pred[i], zero_division=0)
        r = recall_score(y_true[i], y_pred[i], zero_division=0)
        
        if p + r == 0:
            f1 = 0 
        else:
            f1 = 2 * (p * r) / (p + r)
        
        f1_scores.append(f1)
    
    return f1_scores

def save_reports_in_csv(ref_list, gen_list, save_root, save_name):
    assert len(ref_list)==len(gen_list)
    study_ids = [i for i in range(len(ref_list))]
    df_ref = pd.DataFrame({"study_id": study_ids, "report": ref_list})
    df_gen = pd.DataFrame({"study_id": study_ids, "report": gen_list})

    df_ref.to_csv(os.path.join(save_root, f"{save_name}_gt_reports.csv"), index=False)
    df_gen.to_csv(os.path.join(save_root, f"{save_name}_predicted_reports.csv"), index=False)

    print("CSV files created successfully!")


ANATOMICAL_REGIONS = {
    "right lung": 0,
    "right upper lung zone": 1,
    "right mid lung zone": 2,
    "right lower lung zone": 3,
    "right hilar structures": 4,
    "right apical zone": 5,
    "right costophrenic angle": 6,
    "right hemidiaphragm": 7,
    "left lung": 8,
    "left upper lung zone": 9,
    "left mid lung zone": 10,
    "left lower lung zone": 11,
    "left hilar structures": 12,
    "left apical zone": 13,
    "left costophrenic angle": 14,
    "left hemidiaphragm": 15,
    "trachea": 16,
    "spine": 17,
    "right clavicle": 18,
    "left clavicle": 19,
    "aortic arch": 20,
    "mediastinum": 21,
    "upper mediastinum": 22,
    "svc": 23,
    "cardiac silhouette": 24,
    "cavoatrial junction": 25,
    "right atrium": 26,
    "carina": 27,
    "abdomen": 28
}