from chainercv import transforms


class MaskRCNNTransform(object):

    def __init__(self, mask_rcnn, train=True):
        self.mask_rcnn = mask_rcnn
        self.train = train

    def __call__(self, in_data):
        if len(in_data) == 6:
            img, bbox, label, mask, crowd, area = in_data
        elif len(in_data) == 4:
            img, bbox, label, mask = in_data
        else:
            raise ValueError

        img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

        if not self.train:
            if len(in_data) == 6:
                return img, bbox, label, mask, crowd, area
            elif len(in_data) == 4:
                return img, bbox, label, mask
            else:
                raise ValueError

        imgs, sizes, scales = self.mask_rcnn.prepare([img])
        img = imgs[0]
        H, W = sizes[0]
        scale = scales[0]
        _, o_H, o_W = img.shape

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
