import cv2


def view_dataset(dataset, visualize_func=None):
    try:
        assert dataset._transform is not True
    except AttributeError:
        pass

    split = getattr(dataset, 'split', '<unknown>')
    print("Showing dataset '%s' (%d) with split '%s'." %
          (dataset.__class__.__name__, len(dataset), split))

    index = 0
    while True:
        if visualize_func is None:
            viz = dataset[index]
        else:
            viz = visualize_func(dataset, index)
        cv2.imshow(dataset.__class__.__name__, viz[:, :, ::-1])

        k = cv2.waitKey(0)
        if k == ord('q'):
            break
        elif k == ord('n'):
            if index == len(dataset) - 1:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            index += 1
        elif k == ord('p'):
            if index == 0:
                print('WARNING: reached edge index of dataset: %d' % index)
                continue
            index -= 1
