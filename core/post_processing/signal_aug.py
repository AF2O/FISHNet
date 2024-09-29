import torch

def signal_flip(signal_points, img_shape, direction):
    assert signal_points.shape[-1] % 2 == 0
    flipped = signal_points.copy()
    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0] = w - signal_points[..., 0]
    elif direction == 'vertical':
        h = img_shape[0]
        flipped[..., 1] = h - signal_points[..., 1]
    elif direction == 'diagonal':
        w = img_shape[1]
        h = img_shape[0]
        flipped[..., 0] = w - signal_points[..., 0]
        flipped[..., 1] = h - signal_points[..., 1]
    else:
        raise ValueError(f"Invalid flipping direction '{direction}'")
    return flipped

def signal_mapping_back(signals, img_shape, scale_factor, flip, flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    result_signals = []
    for each_roi_signals in signals:
        # new_signal = signal_flip(each_roi_signals, img_shape,
        #                        flip_direction) if flip else signals
        new_signal = each_roi_signals
        new_signal_points = new_signal[:, :2]
        new_signal_labels = new_signal[:, 2][:, None]
        # print('######### new_signal_labels size #########', new_signal_labels.size())
        new_signal_points = new_signal_points.view(-1, 2) / new_signal_points.new_tensor(scale_factor[:2])
        new_signal = torch.cat([new_signal_points, new_signal_labels], dim=-1).numpy()
        result_signals.append(new_signal)
    # print('######### new_signal len #########', len(result_signals), result_signals[0][0].size())
    return result_signals

def get_aug_signals(signals, img_metas):
    signals = signals[0]
    img_info = img_metas
    img_shape = img_info[0]['img_shape']
    scale_factor = img_info[0]['scale_factor']
    flip = img_info[0]['flip']
    flip_direction = img_info[0]['flip_direction']
    signals = signal_mapping_back(signals, img_shape, scale_factor, flip, flip_direction)
    return signals
