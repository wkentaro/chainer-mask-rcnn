from chainercv import transforms


class MaskRCNNTransform(object):

    def __init__(self, mask_rcnn, train=True, is_coco=False):
        self.mask_rcnn = mask_rcnn
        self.train = train
        self._is_coco = is_coco

    def __call__(self, in_data):
        if not self.train and self._is_coco:
            img, bbox, label, mask, crowd, area = in_data
        else:
            img, bbox, label, mask = in_data

        img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

        if not self.train:
            if self._is_coco:
                return img, bbox, label, mask, crowd, area
            else:
                return img, bbox, label, mask

        _, H, W = img.shape
        img = self.mask_rcnn.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        if len(bbox) > 0:
            bbox = transforms.resize_bbox(bbox, (H, W), (o_H, o_W))
        if len(mask) > 0:
            mask = transforms.resize(
                mask, size=(o_H, o_W), interpolation=0)

        # horizontally flip
        img, params = transforms.random_flip(
            img, x_random=True, return_param=True)
        bbox = transforms.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])
        if mask.ndim == 2:
            mask = transforms.flip(
                mask[None, :, :], x_flip=params['x_flip'])[0]
        else:
            mask = transforms.flip(mask, x_flip=params['x_flip'])

        return img, bbox, label, mask, scale
