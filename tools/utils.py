import jsonlines
import json
import cv2
import numpy as np

from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from tools.transform.randaugment import RandomAugment

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def read_jsonl(fname) -> list:
    data = []
    with open(fname, 'r+', encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            data.append(item)
    return data

def save_jsonl(fname, data):
    assert isinstance(data, list)
    with jsonlines.open(fname, mode='w') as f:
        for d in data:
            f.write(d)

def read_json(fname):
    with open(fname, 'r') as f:
        return json.load(f)


def read_image(fname):
    img = cv2.imread(fname)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    return img


def show_box_segms(image, anno):
    width, height = anno['width'], anno['height']
    bboxes = anno['boxes']
    objects = anno['names']
    segments = anno['segms']
    for i, obj in enumerate(objects):
        segments[i][0] = np.array(segments[i][0], np.int32)
        
        x1, y1, x2, y2, s = bboxes[i]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255))
        cv2.putText(image, obj, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 2)
        cv2.polylines(image, [segments[i][0]], True, (255, 0, 255))
        cv2.imshow('image', image)
        cv2.waitKey(0)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def build_transform(augment=False, n_px=224):
    if augment == 'blip_train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(n_px, scale=(0.5, 1.0), interpolation=BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 5, isPIL=True, augs=['Identity', 'AutoContrast', 'Brightness', 'Sharpness', 'Equalize',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif augment == 'blip_test':
        transform = transforms.Compose([
            transforms.Resize((n_px, n_px),
                              interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    elif augment == 'mae_train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(n_px, scale=(0.2, 1.0), interpolation=BICUBIC),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    elif augment == 'mae_test':
        transform = transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    
    return transform


# def build_transform(augment=False, n_px=224):
#     if augment:
#         raise NotImplemented
#     else:
#         return Compose([
#             Resize(n_px, interpolation=BICUBIC),
#             CenterCrop(n_px),
#             _convert_image_to_rgb,
#             ToTensor(),
#             Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
#         ])


# def get_feature(image):
#     # model = fasterrcnn_resnet50_fpn(pretrained=True)
#     backbone = resnet_fpn_backbone('resnet50', pretrained=False)
#     model = FasterRCNN(backbone=backbone, num_classes=91)
#     print([name for name, _ in model.named_children()])

#     images, boxes = torch.rand(4, 3, 600, 800), torch.rand(4, 11, 4)
#     boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
#     labels = torch.randint(1, 91, (4, 11))
#     images = list(image for image in images)
#     targets = []
#     for i in range(len(images)):
#         d = {}
#         d['boxes'] = boxes[i]
#         d['labels'] = labels[i]
#         targets.append(d)
#     output = model(images, targets)
#     new_model = IntermediateLayerGetter(model,  {'roi_heads.box_head.fc7': 'feature'})
#     output = new_model(images)
#     print(output)

#     model.eval()
#     x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#     predictions = model(x)
#     print(predictions)


if __name__ == '__main__':
    train_ori = "baseline/dataset/train-ori.jsonl"
    train_adv = "baseline/dataset/train-adv.jsonl"
    trainset = read_jsonl(train_ori) + read_jsonl(train_adv)
    
    image_path = "baseline/dataset/images/lsmdc_0001_American_Beauty/0001_American_Beauty_00.02.48.867-00.02.55.904@2.jpg"
    anno_path = "baseline/dataset/images/lsmdc_0001_American_Beauty/0001_American_Beauty_00.02.48.867-00.02.55.904@2.json"
    image = read_image(image_path)
    anno = read_json(anno_path)
    # show_box_segms(image, anno)
    # cv2.imshow('image', normalize(image))
    # cv2.waitKey(0)
