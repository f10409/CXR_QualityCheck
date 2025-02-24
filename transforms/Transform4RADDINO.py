import monai as mn
import torch
import numpy as np
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from transformers import AutoImageProcessor
from PIL import Image
from typing import Any, Dict, Optional, Hashable

class RadDINOProcessor(MapTransform):
    """
    A MONAI transform that processes medical images for use with the RAD-DINO vision model.
    
    This transform prepares images by ensuring they have 3 channels, normalizing values,
    and processing them through Microsoft's RAD-DINO image processor to create
    standardized tensor inputs for the model.
    
    Args:
        keys (KeysCollection): Keys to be processed in the data dictionary.
        processor_name (str): Name of the pretrained processor to use from Hugging Face.
        im_size (int): Target image size for processing.
    """
    def __init__(self, keys: KeysCollection, processor_name: str = "microsoft/rad-dino", im_size: int = 518):
        super().__init__(keys)
        self.processor = AutoImageProcessor.from_pretrained(processor_name)
        
        self.processor.size = {"shortest_edge": im_size}    
        self.processor.crop_size = {"height": im_size, "width": im_size}
        
        self.im_size = im_size
        
    def _convert_to_3channel(self, img_array):
        """
        Convert single channel image arrays to 3 channels by repeating the content.
        
        Args:
            img_array (np.ndarray): Input image array.
            
        Returns:
            np.ndarray: 3-channel image array with shape [3,H,W].
        """
        if len(img_array.shape) == 2:  # Single channel [H,W]
            return np.stack([img_array] * 3, axis=0)  # Makes it [3,H,W]
        elif len(img_array.shape) == 3 and img_array.shape[0] == 1:  # [1,H,W]
            return np.repeat(img_array, 3, axis=0)  # Makes it [3,H,W]
        return img_array
    
    def __call__(self, data):
        """
        Process images in the data dictionary.
        
        Args:
            data (Dict): Dictionary containing image data to process.
            
        Returns:
            Dict: Dictionary with processed image tensors.
        """
        d = dict(data)
        for key in self.keys:
            # Get image array
            img_array = d[key]
            
            # Convert to 3 channels if needed
            img_array = self._convert_to_3channel(img_array)
            
            # Convert to [H,W,C] for PIL
            img_array = np.transpose(img_array, (1, 2, 0))
            
            # Normalize to [0,255] if not already
            if img_array.dtype != np.uint8:
                img_array = ((img_array - img_array.min()) / 
                           (img_array.max() - img_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL
            pil_image = Image.fromarray(img_array)
            
            # Process using RAD-DINO processor
            processed = self.processor(images=pil_image, return_tensors="pt")

            # Ensure consistent size
            pixel_values = processed['pixel_values'][0]

            # Verify tensor dimensions
            if pixel_values.shape != (3, self.im_size, self.im_size):
                raise ValueError(f"Unexpected tensor shape: {pixel_values.shape}")
            
            # Store processed tensor
            d[key] = pixel_values
           
        return d


class LoadNumpyArrayd(Transform):
    """
    Dictionary-based transform to load NumPy array from .npy files.
    
    Args:
        keys (KeysCollection): Keys to be processed. Can be a single key or a sequence of keys.
        allow_missing_keys (bool): Don't raise exception if keys are missing.
        dtype (Optional[np.dtype]): Target data type to cast to after loading.
        
    Example:
        >>> transform = LoadNumpyArrayd(keys=["image"])
        >>> data = {"image": "path/to/array.npy"}
        >>> result = transform(data)
        >>> print(result["image"].shape)
    """
    
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        super().__init__()
        self.keys = keys
        self.allow_missing_keys = allow_missing_keys
        self.dtype = dtype

    def __call__(self, data: Dict[Hashable, Any]) -> Dict[Hashable, Any]:
        """
        Args:
            data: Dictionary containing the file paths to .npy files
                
        Returns:
            Dictionary containing the loaded NumPy arrays
        """
        d = dict(data)
        for key in self.keys:
            if key not in d and not self.allow_missing_keys:
                raise KeyError(f"Key {key} not found in data!")
            
            if key in d:
                # Load the NumPy array from file
                try:
                    array = np.load(d[key])
                    if self.dtype is not None:
                        array = array.astype(self.dtype)
                    d[key] = array
                except Exception as e:
                    raise RuntimeError(f"Failed to load NumPy array from {d[key]}: {str(e)}")
                
        return d

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(keys={self.keys}, dtype={self.dtype})"


class Transform4RADDINO:
    """
    A collection of transforms specifically designed for processing images 
    for the RAD-DINO model.
    
    This class creates a composition of MONAI transforms that prepare images
    for inference with the RAD-DINO vision model, handling loading, channel
    adjustments, and tensor conversion.
    
    Args:
        IMG_SIZE (int): Desired spatial size for input images.
    """
    def __init__(self, IMG_SIZE: int):
        """
        Initializes a set of data transformations for RAD-DINO image processing.

        Args:
            IMG_SIZE (int): Desired spatial size for input images.
        """        
        self.predict = mn.transforms.Compose([
            mn.transforms.LoadImageD(keys="img", reader="ITKReader", ensure_channel_first=True),
            mn.transforms.Transposed(keys=["img"], indices=[0, 2, 1]),
            RadDINOProcessor(keys=["img"], im_size=IMG_SIZE),
            mn.transforms.SelectItemsD(keys=["img", "paths"]),
            mn.transforms.ToTensorD(keys="img", dtype=torch.float, track_meta=False)
        ])