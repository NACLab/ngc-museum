"""
Image/tensor patching utility routines.
"""
import numpy as np
from jax import numpy as jnp

class Create_Patches:
    """
    This function will create small patches out of the image based on the provided attributes.

    Args:
        img: jax array of size (H, W)
        patched: (height_patch, width_patch)
        overlap: (height_overlap, width_overlap)

        add_frame: increases the img size by (height_patch - height_overlap, width_patch - width_overlap)
        create_patches: creates small patches out of the image based on the provided attributes.

    Returns:
        jnp.array: Array containing the patches
        shape: (num_patches, patch_height, patch_width)

    """

    def __init__(self, img, patch_shape, overlap_shape):
        self.img = img
        self.height_patch, self.width_patch = patch_shape
        self.height_over, self.width_over = overlap_shape

        self.height, self.width = self.img.shape

        if self.height_over > 0:
            self.nw_patches = (self.width + self.width_over) // (self.width_patch - self.width_over) - 2
        else:
            self.nw_patches = self.width // self.width_patch
        if self.width_over > 0:
            self.nh_patches = (self.height + self.height_over) // (self.height_patch - self.height_over) - 2
        else:
            self.nh_patches = self.height // self.height_patch
        # print('nw_patches', 'nh_patches')
        # print(self.nw_patches, self.nh_patches)
        # print("...........")

    def _add_frame(self):
        """
        This function will add zero frames (increase the dimension) to the image

        Returns:
            image with increased size (x.shape[0], x.shape[1]) -> (x.shape[0] + (height_patch - height_overlap),
                                                                   x.shape[1] + (width_patch - width_overlap))
        """
        self.img = np.hstack((jnp.zeros((self.img.shape[0], (self.height_patch - self.height_over))),
                              self.img,
                              jnp.zeros((self.img.shape[0], (self.height_patch - self.height_over)))))
        self.img = np.vstack((jnp.zeros(((self.width_patch - self.width_over), self.img.shape[1])),
                              self.img,
                              jnp.zeros(((self.width_patch - self.width_over), self.img.shape[1]))))

        self.height, self.width = self.img.shape

        self.nw_patches = (self.width + self.width_over) // (self.width_patch - self.width_over) - 2
        self.nh_patches = (self.height + self.height_over) // (self.height_patch - self.height_over) - 2



    def create_patches(self, add_frame=False, center=True):
        """
        This function will create small patches out of the image based on the provided attributes.

        Keyword Args:
            add_frame: If true the function will add zero frames (increase the dimension) to the image

        Returns:
            jnp.array: Array containing the patches
            shape: (num_patches, patch_height, patch_width)
        """

        if add_frame == True:
            self._add_frame()

        if center == True:
            mu = np.mean(self.img, axis=0, keepdims=True)
            self.img  = self.img - mu

        result = []
        for nh_ in range(self.nh_patches):
            for nw_ in range(self.nw_patches):
                img_ = self.img[(self.height_patch - self.height_over) * nh_: nh_ * (
                            self.height_patch - self.height_over) + self.height_patch
                , (self.width_patch - self.width_over) * nw_: nw_ * (
                            self.width_patch - self.width_over) + self.width_patch]

                if img_.shape == (self.height_patch, self.width_patch):
                    result.append(img_)
        return jnp.array(result)