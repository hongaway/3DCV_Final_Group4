# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from eval import TorchEval
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer

from scipy.spatial.transform import Rotation as R

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.orien_embed  = MLP(hidden_dim, hidden_dim, 6 ,3)
        self.trans_embed = MLP(hidden_dim, hidden_dim,3,3)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        outputs_orien = self.orien_embed(hs).sigmoid()
        outputs_trans = self.trans_embed(hs)
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], 'pred_orien': outputs_orien[-1], 'pred_trans': outputs_trans[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, cld):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.cld = cld
        self.teval = TorchEval()
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not h
            #line = line[:-1]
            losses['class_error'] = 89
            pass
        #pointxyz = np.loadtxt('./YCB_Video_Dataset/models/'+ line +'/points.xyz', dtype=np.float32)
        #cld.append(pointxyz)
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    def martix_to_qu(self, matrix):
        r = R.from_matrix(matrix)
        print(r.as_quat())
        
    def cal_add_cuda(self, pred_RT, gt_RT, p3ds):
        pred_p3ds = torch.mm(p3ds, pred_RT.transpose(1, 0).float())
        gt_p3ds = torch.mm(p3ds, gt_RT.transpose(1, 0)) 
        dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
        return torch.mean(dis)

    def cal_adds_cuda(self, pred_RT, gt_RT, p3ds):
        N, _ = p3ds.size()
        pd = torch.mm(p3ds, pred_RT.transpose(1, 0).float()) 
        pd = pd.view(1, N, 3).repeat(N, 1, 1)
        gt = torch.mm(p3ds, gt_RT.transpose(1, 0)) 
        gt = gt.view(N, 1, 3).repeat(1, N, 1)
        dis = torch.norm(pd - gt, dim=2)
        mdis = torch.min(dis, dim=1)[0]
        return torch.mean(mdis)


    def rotation_6d_to_matrix(self, rot_6d):
        """
        Given a 6D rotation output, calculate the 3D rotation matrix in SO(3) using the Gramm Schmit process
        For details: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
        """
        # bs, n_q, _ = rot_6d.shape
        # rot_6d = rot_6d.view(-1, 6)
        m1 = rot_6d[:, 0:3]
        m2 = rot_6d[:, 3:6]

        x = F.normalize(m1, p=2, dim=1)
        z = torch.cross(x, m2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        y = torch.cross(z, x, dim=1)
        rot_matrix = torch.cat((x.view(-1, 3, 1), y.view(-1, 3, 1), z.view(-1, 3, 1)), 2)  # Rotation Matrix lying in the SO(3)
        rot_matrix = rot_matrix.view(rot_6d.shape[0], 3, 3)  #.transpose(2, 3)
        return rot_matrix

    # def loss_poses(self, outputs, targets, indices, num_boxes):
 
   
      
    #     idx = self._get_src_permutation_idx(indices)
    
    #     src_orien = outputs['pred_orien'][idx]
    #     src_trans = outputs['pred_trans'][idx]
        
        

    #     p3d = []
    #     target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    #     for id in target_classes_o:
    #         ranIdx = np.random.choice(2620, 1500,replace= False)
    #         p3d.append(self.cld[id - 1][ranIdx])

    #     target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)
    #     src_rotMat = []
    #     #print(src_orien.shape)
    #     for orien in src_orien:
    #         orien_np = orien.detach().cpu().numpy()
    #         r = R.from_quat(orien_np).as_matrix()
    #         src_rotMat.append(r)
    #     p3d = np.array(p3d)
    #     p3d = torch.Tensor(p3d).cuda()
    #     src_rotMat = np.array(src_rotMat)
    #     src_rotMat = torch.Tensor(src_rotMat).cuda()
    #     #print(target_classes_o.shape)
    #     # target_quan = []
    #     # for orien in target_orien:
    #     #     orien_np = orien.detach().cpu().numpy()
    #     #     r = R.from_matrix(orien_np).as_quat()
    #     #     target_quan.append(r)
    #     # target_quan = np.array(target_quan)
    #     # target_quan = torch.Tensor(target_quan).cuda()

    #     target_trans = torch.cat([t['RTs'][i][:, :, 3] for t, (_, i) in zip(targets, indices)], dim=0)
    #     loss_orien = 0
    #     for i in range(src_orien.shape[0]):
    #         if(target_classes_o[i] in [13, 16, 19, 20, 21]):
    #             loss_orien +=self.cal_adds_cuda(src_rotMat[i], target_orien[i], p3d[i])
    #         else:
    #             loss_orien += self.cal_add_cuda(src_rotMat[i], target_orien[i], p3d[i])
    #         #loss_orien += 2 - 2 *(src_orien[i].dot(target_quan[i])) ** 2

    #     L = torch.nn.MSELoss(reduction='sum')
    #     loss_trans = L(src_trans, target_trans)
    #     losses = {}
    #     losses['loss_orien'] = loss_orien.sum() / src_trans.shape[0]
    #     losses['loss_trans'] = loss_trans.sum() / src_trans.shape[0]
        
    #     return losses

    # def loss_poses(self, outputs, targets, indices, num_boxes):
    
    #         idx = self._get_src_permutation_idx(indices)
        
    #         src_orien = outputs['pred_orien'][idx]
    #         src_trans = outputs['pred_trans'][idx]
            
    #         # print(src_orien.shape)
    #         # print("fucklkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk")
    #         # print(src_trans.shape)

    #         ###### edit
    #         target_quan = []
    #         target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)
    #         for orien in target_orien:
    #             orien_np = orien.detach().cpu().numpy()
    #             r = R.from_matrix(orien_np).as_quat()
    #             target_quan.append(r)
    #         target_quan = np.array(target_quan)
    #         target_quan = torch.Tensor(target_quan).cuda()
    #         eps = 1e-4
    #         dot_product = torch.mul(src_orien, target_quan)
    #         dp_sum = torch.sum(dot_product, 1)
    #         dp_square = torch.square(dp_sum)
    #         loss_quat = - torch.log(dp_square + eps)
    #         ###### edit
    #         target_trans = torch.cat([t['RTs'][i][:, :, 3] for t, (_, i) in zip(targets, indices)], dim=0)
    
    #         loss_trans = F.l1_loss(src_trans, target_trans, reduction='none')
    #         losses = {}
    #         losses['loss_orien'] = loss_quat.sum() / src_orien.shape[0]
    #         losses['loss_trans'] = loss_trans.sum() / src_trans.shape[0]
    #         return losses

    def loss_poses(self, outputs, targets, indices, num_boxes):
        eps = 1e-6
        idx = self._get_src_permutation_idx(indices)
        src_orien = outputs["pred_orien"][idx]
        src_trans = outputs['pred_trans'][idx]
        # print("src_orien", src_orien.shape)
        # print("src_trans", src_trans.shape)
        src_rot = self.rotation_6d_to_matrix(src_orien)
        # print("src_rot", src_rot.shape)
    
        target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)
        
        n_obj = len(target_orien)

        product = torch.bmm(src_rot, target_orien.transpose(1, 2))
        trace = torch.sum(product[:, torch.eye(3).bool()], 1)
        theta = torch.clamp(0.5 * (trace - 1), -1 + eps, 1 - eps)
        rad = torch.acos(theta)
        
        target_trans = torch.cat([t['relative_position'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_trans = F.mse_loss(src_trans, target_trans, reduction='none')
        loss_trans = torch.sum(loss_trans, dim=1)
        loss_trans = torch.sqrt(loss_trans)
        
        losses = {}
        losses["loss_orien"] = rad.sum() / n_obj
        losses['loss_trans'] = loss_trans.sum() / src_trans.shape[0]
        return losses


       
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'poses': self.loss_poses
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    def evals(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
    
        
        src_orien = outputs['pred_orien'][idx]
        # src_orien = quaternion_to_matrix(src_orien)
        src_orien = self.rotation_6d_to_matrix(src_orien)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        
        src_trans = outputs['pred_trans'][idx]
        source_RTs = torch.cat([ torch.cat((orien,trans.view(3,-1)), dim= 1 )for (orien, trans) in zip(src_orien, src_trans)], dim= 0).view(-1,3,4)
        target_RTs = torch.cat([t['RTs'][i][:, :, :] for t, (_, i) in zip(targets, indices)], dim=0)

        
        #target_orien = torch.cat([t['RTs'][i][:, :, :3] for t, (_, i) in zip(targets, indices)], dim=0)
        #target_trans = torch.cat([t['RTs'][i][:, :, 3] for t, (_, i) in zip(targets, indices)], dim=0)
        
        self.teval.eval_pose_parallel(target_classes_o, source_RTs, target_classes_o, target_RTs, )
    def forward(self, outputs, targets, eval = False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if loss =='poses':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        if eval == True:
            self.evals(outputs, targets, indices)
            
        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def load_model_pcd():
    cld = []
    f = open('/home/ai2lab/Desktop/台達計畫/T6D_implement/T6D-direct/YCB_Video_Dataset/image_sets/classes.txt')
    for line in f.readlines():
        if '\n' in line:
            line = line[:-1]
        pointxyz = np.loadtxt('./YCB_Video_Dataset/models/'+ line +'/points.xyz', dtype=np.float32)
        cld.append(pointxyz)
    return cld
        
def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 20 if args.dataset_file != 'coco' else 91
    num_classes = 21 if args.dataset_file == 'YCB_V' else num_classes
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    cld = load_model_pcd()
 
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    weight_dict['loss_orien'] = 1
    weight_dict['loss_trans'] = 1
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality', 'poses']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses, cld=cld)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
