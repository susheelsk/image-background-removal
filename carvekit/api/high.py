"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.u2net import U2NET
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.trimap.generator import TrimapGenerator


class HiInterface(Interface):
    def __init__(self, batch_size_seg=5, batch_size_matting=1,
                 device='cpu', seg_mask_size=320, matting_mask_size=2048):
        """
        Initializes High Level interface.

        Args:
            matting_mask_size:  The size of the input image for the matting neural network.
            seg_mask_size: The size of the input image for the segmentation neural network.
            batch_size_seg: Number of images processed per one segmentation neural network call.
            batch_size_matting: Number of images processed per one matting neural network call.
            device: Processing device

        Notes:
            Changing seg_mask_size may cause an out-of-memory error if the value is too large, and it may also
            result in reduced precision. I do not recommend changing this value. You can change matting_mask_size in
            range from (1024 to 4096) to improve object edge refining quality, but it will cause extra large RAM and
            video memory consume. Also, you can change batch size to accelerate background removal, but it also causes
            extra large video memory consume, if value is too big.
        """
        self.u2net = U2NET(device=device, batch_size=batch_size_seg, input_image_size=seg_mask_size)
        self.fba = FBAMatting(batch_size=batch_size_matting, device=device, input_tensor_size=matting_mask_size)
        self.trimap_generator = TrimapGenerator()
        super(HiInterface, self).__init__(pre_pipe=None,
                                          seg_pipe=self.u2net,
                                          post_pipe=MattingMethod(matting_module=self.fba,
                                                                  trimap_generator=self.trimap_generator,
                                                                  device=device),
                                          device=device)