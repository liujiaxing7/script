import torch.nn as nn
import torch.nn.functional as F
import torch

from to_onnx.ssd.utils import box_utils

ALPHA = 0.25
GAMMA = 2

def onehot_embedding(labels, channels):
    bin_labels = labels.new_full((labels.size(0), channels), 0)
    id_objects = torch.nonzero(labels >= 1).squeeze()
    if id_objects.numel() > 0:
        id_label = labels[id_objects]
        bin_labels[id_objects, id_label] = 1

    return bin_labels

def expand(w, num):
    if w is None:
        w = None
    else:
        w = w.view(-1, 1).expand(w.size(0), num)

    return w

class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos_ratio, use_sigmoid=False, focal_loss=False):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.use_sigmoid = use_sigmoid
        self.focal_loss = focal_loss
        self.background_id = 0

    def _background_loss(self, confidence):
        with torch.no_grad():
            if self.use_sigmoid:
                loss = -F.logsigmoid(confidence)[:, :, self.background_id]
            else:
                loss = -F.log_softmax(confidence, dim=2)[:, :, self.background_id]

        return loss

    def _valid_anchors(self, confidence, labels):
        background_loss = self._background_loss(confidence)
        mask = box_utils.hard_negative_mining(background_loss, labels, self.neg_pos_ratio)

        return mask

    def _focal_loss(self, x, y):
        assert x.shape == y.shape
        xt = x*(2*x+1)
        pt = (2*xt-1).sigmoid()
        return (-pt.log()/2).sum()

    def _class_loss(self, confidence, labels, weight=None):
        num_classes = confidence.size(1)
        confidence = confidence.view(-1, num_classes)
        if self.use_sigmoid:
            labels_onehot = labels
            if confidence.dim() != labels.dim():
                labels_onehot = onehot_embedding(labels, confidence.size(-1))

            if self.focal_loss:
                classification_loss = self._focal_loss(confidence, labels_onehot).float()

            else:
                classification_loss = F.binary_cross_entropy_with_logits(confidence, labels_onehot.float(), None, reduction='sum')
        else:
            classification_loss = F.cross_entropy(confidence, labels, reduction='sum')

        return classification_loss

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        # import ipdb
        # ipdb.set_trace()
        num_classes = confidence.size(2)
        mask = self._valid_anchors(confidence, labels)
        classification_loss = self._class_loss(confidence[mask, :], labels[mask])

        pos_mask = labels > 0
        if torch.any(pos_mask):
            predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
            gt_locations = gt_locations[pos_mask, :].view(-1, 4)
            smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
            num_pos = gt_locations.size(0)
        else:
            smooth_l1_loss=0
            num_pos = 1
        return smooth_l1_loss / num_pos, classification_loss / num_pos
