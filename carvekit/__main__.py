from pathlib import Path

import click
import tqdm

from carvekit.utils.image_utils import ALLOWED_SUFFIXES
from carvekit.utils.pool_utils import batch_generator, thread_pool_processing
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from carvekit.utils.fs_utils import save_file


@click.command('removebg', help="Performs background removal on specified photos using console interface.")
@click.option('-i', required=True, type=str, help='Path to input file or dir')
@click.option('-o', default="none", type=str, help="Path to output file or dir")
@click.option('--pre', default='none', type=str, help='Preprocessing method')
@click.option('--post', default='fba', type=str, help='Postprocessing method.')
@click.option('--net', default='u2net', type=str, help='Segmentation Network')
@click.option('--recursive', default=False, type=bool, help='Enables recursive search for images in a folder')
@click.option('--batch_size', default=10, type=int, help='Batch Size for list of images to be loaded to RAM')
@click.option('--batch_size_seg', default=5, type=int,
              help='Batch size for list of images to be processed by segmentation '
                   'network')
@click.option('--batch_size_mat', default=1, type=int, help='Batch size for list of images to be processed by matting '
                                                            'network')
@click.option('--seg_mask_size', default=320, type=int,
              help='The size of the input image for the segmentation neural network.')
@click.option('--matting_mask_size', default=2048, type=int,
              help='The size of the input image for the matting neural network.')
@click.option('--device', default="cpu", type=str,
              help='Processing Device.')
def removebg(i: str, o: str, pre: str, post: str, net: str, recursive: bool,
             batch_size: int, batch_size_seg: int, batch_size_mat: int, seg_mask_size: int,
             matting_mask_size: int,
             device: str):
    out_path = Path(o)
    input_path = Path(i)
    if input_path.is_dir():
        if recursive:
            all_images = input_path.glob("*.*")
        else:
            all_images = input_path.rglob("*.*")
        all_images = [i for i in all_images if i.suffix.lower() in ALLOWED_SUFFIXES
                      and '_bg_removed' not in i.name]
    else:
        all_images = [input_path]

    interface_config = MLConfig(
        segmentation_network=net,
        preprocessing_method=pre,
        postprocessing_method=post,
        device=device,
        batch_size_seg=batch_size_seg,
        batch_size_matting=batch_size_mat,
        seg_mask_size=seg_mask_size,
        matting_mask_size=matting_mask_size
    )

    interface = init_interface(interface_config)
    images_without_background = []

    for image_batch in tqdm.tqdm(batch_generator(all_images, n=batch_size),
                                 total=int(len(all_images) / batch_size),
                                 desc="Removing background", unit=" image batch",
                                 colour="blue"):
        images_without_background = interface(image_batch)  # Remove background
        thread_pool_processing(lambda x: save_file(out_path, image_batch[x], images_without_background[x]),
                               range((len(image_batch))))  # Drop images to fs


if __name__ == "__main__":
    removebg()
