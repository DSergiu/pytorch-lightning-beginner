import os

import matplotlib.pyplot as plt


def load_trained_model_for_evaluation(model_name, version_no):
    test_version_dir = './lightning_logs/version_' + str(version_no)
    hparams_path = os.path.join(test_version_dir, 'hparams.yaml')
    ckpt_dir_path = os.path.join(test_version_dir, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir_path, os.listdir(ckpt_dir_path)[0])
    model = model_name.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        hparams_file=hparams_path,
        map_location=None
    )
    model.setup(None) # Load datasets if any
    model.eval()      # Configure evaluation mode
    return model


def plot_images(imgs: list, figsize=None) -> None:
    n = len(imgs)
    f = plt.figure(figsize=figsize)
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.imshow(imgs[i], cmap='gray')
    plt.show(block=True)
