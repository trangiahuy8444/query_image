# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import torch
import torchvision.transforms as T
from PIL import Image
import json
import matplotlib.pyplot as plt

# Các lớp và mô hình
from RelTR.models.backbone import Backbone, Joiner
from RelTR.models.position_encoding import PositionEmbeddingSine
from RelTR.models.transformer import Transformer
from RelTR.models.reltr import RelTR

# Xử lý ảnh
import requests
from PIL import Image
from io import BytesIO
import os

CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

def load_model(checkpoint_path='./ckpt/checkpoint0149.pth'):
    # Thiết lập model một lần
    position_embedding = PositionEmbeddingSine(128, normalize=True)
    backbone = Backbone('resnet50', False, False, False)
    backbone = Joiner(backbone, position_embedding)
    backbone.num_channels = 2048

    transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                              dim_feedforward=2048,
                              num_encoder_layers=6,
                              num_decoder_layers=6,
                              normalize_before=False,
                              return_intermediate_dec=True)

    model = RelTR(backbone, transformer, num_classes=151, num_rel_classes=51,
                  num_entities=100, num_triplets=200)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model

# Tiền xử lý ảnh
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Hàm xử lý bbox
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

# Hàm dự đoán với đầu vào là ảnh
def predict(image, model):
    im = Image.open(image) 
    # im = processImage(image)
    img = transform(im).unsqueeze(0)
    
    # Chuyển input sang cùng device với model
    device = next(model.parameters()).device
    img = img.to(device)
    
    # propagate through the model
    outputs = model(img)

    # keep only predictions with >0.3 confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)
    
    topk = 10 # display up to 10 images
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]
        
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        predictions = []
        sameCheck = True
        for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, sub_bboxes_scaled, obj_bboxes_scaled):
            
            subject_class = CLASSES[probas_sub[idx].argmax()]
            relation_class = REL_CLASSES[probas[idx].argmax()]
            object_class = CLASSES[probas_obj[idx].argmax()]

            # Construct prediction dictionary
            prediction = {
                "subject": {
                    "class": subject_class,
                    "bbox": torch.tensor([sxmin, symin, sxmax, symax], device=device),
                    "score": probas_sub[idx].max()
                },
                "relation": {
                    "class": relation_class,
                    "score": probas[idx].max()
                },
                "object": {
                    "class": object_class,
                    "bbox": torch.tensor([oxmin, oymin, oxmax, oymax], device=device),
                    "score": probas_obj[idx].max()
                }
            }
            predictions.append(prediction)

    return predictions