# ------------------------------------------------------------------------
# DAHOI
# Copyright (c) 2022 Shuailei Ma. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QAHOI (https://github.com/cjw2021/QAHOI)
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse
from pathlib import Path
import numpy as np
import copy
import pickle
import sys, os

sys.path.insert(0, os.path.join(os.getcwd(), "models/ops/lib/python3.8/site-packages/MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg"))  # if you install DCNv2 in its dictory

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import build_dataset

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.backbone import build_backbone
from models.deformable_transformer import build_deforamble_transformer
from models.QAHOI import QAHOI


class PostProcessHOI(nn.Module):

    def __init__(self, subject_category_id, correct_mat, no_obj=False, use_nms=True):
        super().__init__()
        self.subject_category_id = subject_category_id
        self.no_obj = no_obj
        self.use_nms = use_nms
        self.nms_thresh = 0.5

        correct_mat = np.concatenate((correct_mat, np.ones((correct_mat.shape[0], 1))), axis=1)
        self.register_buffer('correct_mat', torch.from_numpy(correct_mat))

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_obj_logits'], \
                                                                        outputs['pred_verb_logits'], \
                                                                        outputs['pred_sub_boxes'], \
                                                                        outputs['pred_obj_boxes']



        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        verb_scores = out_verb_logits.sigmoid()
        num_verb_classes = verb_scores.shape[-1]
        
        # top 100
        obj_prob_class_all = obj_prob[:, :, :-1] if self.no_obj else obj_prob
        num_obj_classes = obj_prob_class_all.shape[-1]

        topk_values, topk_indexes = torch.topk(obj_prob_class_all.flatten(1), 100, dim=1)
        obj_scores = topk_values
        topk_boxes = topk_indexes // num_obj_classes
        obj_labels = topk_indexes % num_obj_classes

        # top 100
        verb_scores = torch.gather(verb_scores, 1, topk_boxes.unsqueeze(-1).repeat(1,1,num_verb_classes))
        out_obj_boxes = torch.gather(out_obj_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        out_sub_boxes = torch.gather(out_sub_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            bboxes = [{'bbox': bbox, 'category_id': label} for bbox, label in zip(b.to('cpu').numpy(), l.to('cpu').numpy())]

            hoi_scores = vs * os.unsqueeze(1)

            verb_labels = torch.arange(hoi_scores.shape[1], device=self.correct_mat.device).view(1, -1).expand(
                hoi_scores.shape[0], -1)
            object_labels = ol.view(-1, 1).expand(-1, hoi_scores.shape[1])
            masks = self.correct_mat[verb_labels.reshape(-1), object_labels.reshape(-1)].view(hoi_scores.shape)
            hoi_scores *= masks

            ids = torch.arange(b.shape[0])

            hois = [{'subject_id': subject_id, 'object_id': object_id, 'category_id': category_id, 'score': score} for
                    subject_id, object_id, category_id, score in zip(ids[:ids.shape[0] // 2].to('cpu').numpy(),
                                                                     ids[ids.shape[0] // 2:].to('cpu').numpy(),
                                                                     verb_labels.to('cpu').numpy(), hoi_scores.to('cpu').numpy())]

            result_t = {
                'predictions': bboxes,
                'hoi_prediction': hois
            }
            
            if self.use_nms:
                result_t = self.triplet_nms_filter(result_t)

            results.append(result_t)

        return results

    def triplet_nms_filter(self, preds):
        pred_bboxes = preds['predictions']
        pred_hois = preds['hoi_prediction']
        all_triplets = {}
        for index, pred_hoi in enumerate(pred_hois):
            triplet = str(pred_bboxes[pred_hoi['subject_id']]['category_id']) + '_' + \
                      str(pred_bboxes[pred_hoi['object_id']]['category_id']) + '_' + str(pred_hoi['category_id'])

            if triplet not in all_triplets:
                all_triplets[triplet] = {'subs':[], 'objs':[], 'scores':[], 'indexes':[]}
            all_triplets[triplet]['subs'].append(pred_bboxes[pred_hoi['subject_id']]['bbox'])
            all_triplets[triplet]['objs'].append(pred_bboxes[pred_hoi['object_id']]['bbox'])
            all_triplets[triplet]['scores'].append(pred_hoi['score'])
            all_triplets[triplet]['indexes'].append(index)

        all_keep_inds = []
        for triplet, values in all_triplets.items():
            subs, objs, scores = values['subs'], values['objs'], values['scores']
            keep_inds = self.pairwise_nms(np.array(subs), np.array(objs), np.array(scores))

            keep_inds = list(np.array(values['indexes'])[keep_inds])
            all_keep_inds.extend(keep_inds)

        preds_filtered = {
            'predictions': pred_bboxes,
            'hoi_prediction': list(np.array(preds['hoi_prediction'])[all_keep_inds])
            }

        return preds_filtered

    def pairwise_nms(self, subs, objs, scores):
        sx1, sy1, sx2, sy2 = subs[:, 0], subs[:, 1], subs[:, 2], subs[:, 3]
        ox1, oy1, ox2, oy2 = objs[:, 0], objs[:, 1], objs[:, 2], objs[:, 3]

        sub_areas = (sx2 - sx1 + 1) * (sy2 - sy1 + 1)
        obj_areas = (ox2 - ox1 + 1) * (oy2 - oy1 + 1)

        max_scores = np.max(scores, axis=1)
        order = max_scores.argsort()[::-1]

        keep_inds = []
        while order.size > 0:
            i = order[0]
            keep_inds.append(i)

            sxx1 = np.maximum(sx1[i], sx1[order[1:]])
            syy1 = np.maximum(sy1[i], sy1[order[1:]])
            sxx2 = np.minimum(sx2[i], sx2[order[1:]])
            syy2 = np.minimum(sy2[i], sy2[order[1:]])

            sw = np.maximum(0.0, sxx2 - sxx1 + 1)
            sh = np.maximum(0.0, syy2 - syy1 + 1)
            sub_inter = sw * sh
            sub_union = sub_areas[i] + sub_areas[order[1:]] - sub_inter

            oxx1 = np.maximum(ox1[i], ox1[order[1:]])
            oyy1 = np.maximum(oy1[i], oy1[order[1:]])
            oxx2 = np.minimum(ox2[i], ox2[order[1:]])
            oyy2 = np.minimum(oy2[i], oy2[order[1:]])

            ow = np.maximum(0.0, oxx2 - oxx1 + 1)
            oh = np.maximum(0.0, oyy2 - oyy1 + 1)
            obj_inter = ow * oh
            obj_union = obj_areas[i] + obj_areas[order[1:]] - obj_inter

            ovr = sub_inter/sub_union * obj_inter/obj_union
            inds = np.where(ovr <= self.nms_thresh)[0]

            order = order[inds + 1]
        return keep_inds

def get_args_parser():
    parser = argparse.ArgumentParser('Dynamic Anchor for Human Object Interaction Detection', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lr_drop', default=120, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--nms_thresh', default=0.5, type=float)
    parser.add_argument('--use_checkpoint', action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--cfg', type=str, default='configs/dat_base.yaml', metavar="FILE",
                        help='path to config file', )
    parser.add_argument('--backbone', default='dat_base', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--N', default=5, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # HOI
    parser.add_argument('--pretrained', type=str, default="",
                        help='Pretrained model path')
    parser.add_argument('--num_verb_classes', type=int, default=117,
                        help="Number of verb classes")
    parser.add_argument('--num_obj_classes', type=int, default=80,
                        help="Number of object classes")
    parser.add_argument('--subject_category_id', default=0, type=int)

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=2.5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=1, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_obj_class', default=1, type=float,
                        help="Object class coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=1, type=float,
                        help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--obj_loss_coef', default=1, type=float)
    parser.add_argument('--verb_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='hico')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--hoi_path', default='/home/msl/hico_20160224_det/hico_20160224_det/', type=str)

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_extra', action='store_true')
    parser.add_argument('--use_nms', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    return parser

def main(args, config):
    print("git:\n  {}\n".format(utils.get_sha()))

    print(args)

    valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                     14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                     24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                     37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                     58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                     72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                     82, 84, 85, 86, 87, 88, 89, 90)

    verb_classes = ['hold_obj', 'stand', 'sit_instr', 'ride_instr', 'walk', 'look_obj', 'hit_instr', 'hit_obj',
                    'eat_obj', 'eat_instr', 'jump_instr', 'lay_instr', 'talk_on_phone_instr', 'carry_obj',
                    'throw_obj', 'catch_obj', 'cut_instr', 'cut_obj', 'run', 'work_on_computer_instr',
                    'ski_instr', 'surf_instr', 'skateboard_instr', 'smile', 'drink_instr', 'kick_obj',
                    'point_instr', 'read_obj', 'snowboard_instr']

    device = torch.device(args.device)

    dataset_val = build_dataset(image_set='val', args=args)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    args.lr_backbone = 0
    args.masks = False
    backbone = build_backbone(args, config)
    transformer = build_deforamble_transformer(args)
    model = QAHOI(
        backbone,
        transformer,
        num_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        no_obj=args.no_obj
    )
    postprocess = PostProcessHOI(args.subject_category_id, dataset_val.correct_mat, no_obj=args.no_obj, use_nms=args.use_nms)

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    postprocess.to(device)

    detections = generate(model, postprocess, data_loader_val, device, verb_classes, args.missing_category_id)

    with open(args.save_path, 'wb') as f:
        pickle.dump(detections, f, protocol=2)


@torch.no_grad()
def generate(model, post_processor, data_loader, device, verb_classes, missing_category_id):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate:'

    detections = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = post_processor(outputs, orig_target_sizes)

        for img_results, img_targets in zip(results, targets):
            for hoi in img_results['hoi_prediction']:
                detection = {
                    'image_id': img_targets['img_id'],
                    'person_box': img_results['predictions'][hoi['subject_id']]['bbox'].tolist()
                }
                if img_results['predictions'][hoi['object_id']]['category_id'] == missing_category_id:
                    object_box = [np.nan, np.nan, np.nan, np.nan]
                else:
                    object_box = img_results['predictions'][hoi['object_id']]['bbox'].tolist()
                cut_agent = 0
                hit_agent = 0
                eat_agent = 0
                for idx, score in zip(hoi['category_id'], hoi['score']):
                    verb_class = verb_classes[idx]
                    score = score.item()
                    if len(verb_class.split('_')) == 1:
                        detection['{}_agent'.format(verb_class)] = score
                    elif 'cut_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        cut_agent = score if score > cut_agent else cut_agent
                    elif 'hit_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        hit_agent = score if score > hit_agent else hit_agent
                    elif 'eat_' in verb_class:
                        detection[verb_class] = object_box + [score]
                        eat_agent = score if score > eat_agent else eat_agent
                    else:
                        detection[verb_class] = object_box + [score]
                        detection['{}_agent'.format(
                            verb_class.replace('_obj', '').replace('_instr', ''))] = score
                detection['cut_agent'] = cut_agent
                detection['hit_agent'] = hit_agent
                detection['eat_agent'] = eat_agent
                detections.append(detection)

    return detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    config = get_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)
    