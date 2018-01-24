from chainercv import transforms


def flip_image(image, x_flip=False, y_flip=False):
    # image has tensor size of (C, H, W)
    if y_flip:
        image = image[:, ::-1, :]
    if x_flip:
        image = image[:, :, ::-1]
    return image


class MaskRCNNTransform(object):

    def __init__(self, mask_rcnn, train=True):
        self.mask_rcnn = mask_rcnn
        self.train = train

    def __call__(self, in_data):
        img, bbox, label, mask = in_data
        img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

        if not self.train:
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
            mask = flip_image(mask[None, :, :], x_flip=params['x_flip'])[0]
        else:
            mask = flip_image(mask, x_flip=params['x_flip'])

        return img, bbox, label, mask, scale
