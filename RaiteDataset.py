# import os
# from typing import Any, Callable, List, Optional, Tuple

# import torch
# import torchvision.transforms.functional as TF
# from torchvision.datasets import CocoDetection
# from PIL import ImageFile

# ImageFile.LOAD_TRUNCATED_IMAGES = True

# class RaiteDataset(CocoDetection):
#     """RAITE 2023 Dataset.

#     Args:
#         root (string): Root directory where images are downloaded to.
#         annFile (string): Path to json annotation file.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.PILToTensor``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         transforms (callable, optional): A function/transform that takes input sample and its target as entry
#             and returns a transformed version.
#     """

#     def __init__(
#         self,
#         root: str,
#         annFile: str,
#         transform: Optional[Callable] = None,
#         target_transform: Optional[Callable] = None,
#         transforms: Optional[Callable] = None,
#     ) -> None:
#         super().__init__(root, annFile, transforms, transform, target_transform)

#     @staticmethod
#     def _raite_image_transform(image):
#         return TF.convert_image_dtype(TF.pil_to_tensor(image), torch.float)

#     @staticmethod
#     def _raite_target_transform(target):
#         boxes = [x['bbox'] for x in target]
#         boxes = torch.as_tensor(boxes).reshape(-1, 4)
#         boxes[:,2:] += boxes[:,:2]

#         labels = [x['category_id'] for x in target]
#         labels = torch.tensor(labels, dtype=torch.int64)

#         target = {}
#         target['boxes'] = boxes
#         target['labels'] = labels

#         return target

#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         id = self.ids[index]
#         image = self._load_image(id)
#         target = self._load_target(id)

#         image = RaiteDataset._raite_image_transform(image)
#         target = RaiteDataset._raite_target_transform(target)

#         if self.transforms is not None:
#             image, target = self.transforms(image, target)

#         return image, target

import os
from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision.transforms.v2.functional as TF
import torchvision.datapoints
from torchvision.datasets import CocoDetection
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class RaiteDataset(CocoDetection):
    """RAITE 2023 Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        resize: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, annFile, transforms, transform, target_transform)
        self.resize = resize

    @staticmethod
    def _raite_image_transform(image):
        return TF.convert_image_dtype(TF.pil_to_tensor(image), torch.float)

    @staticmethod
    def _raite_target_transform(target):
        boxes = [x['bbox'] for x in target]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        boxes[:,2:] += boxes[:,:2]

        labels = [x['category_id'] for x in target]
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels

        return target

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        image = RaiteDataset._raite_image_transform(image)
        target = RaiteDataset._raite_target_transform(target)

        target['boxes'] =  torchvision.datapoints.BoundingBox(target['boxes'],format="xyxy",spatial_size=image.shape[-2:])

        if self.transforms is not None:
            # image = self.transforms(image)
            # target['boxes'] = self.transforms(target['boxes'])
            image, target = self.transforms(image, target)
            # target["boxes"] = self.transforms(target["boxes"])
        if self.resize is not None:
            # image =  self.resize(image)
            # target["boxes"] = self.resize(target["boxes"])
            image,target = self.resize(image,target)

        return image, target