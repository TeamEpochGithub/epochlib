import torch

from epochalyst.training.augmentation import image_augmentations


class TestImageAugmentations:
    def test_cutmix(self):
        # Create a CutMix instance
        cutmix = image_augmentations.CutMix(p=1.0)

        # Create dummy input and labels
        x = torch.cat(
            [torch.ones(16, 1, 100, 100), torch.zeros(16, 1, 100, 100)], dim=0
        )
        # Multiclass labels
        y = torch.cat([torch.ones(16, 2), torch.zeros(16, 2)], dim=0)
        # Apply CutMix augmentation
        augmented_x, augmented_y = cutmix(x, y)

        # Assert the output shapes are correct
        assert augmented_x.shape == x.shape
        assert augmented_y.shape == y.shape

        # Because the images are all ones and zeros the mean of the pixels should be equal to the labels after being transformed
        assert torch.allclose(augmented_x.mean(dim=-1).mean(dim=-1), augmented_y)

        cutmix = image_augmentations.CutMix(p=0)
        augmented_x, augmented_y = cutmix(x, y)

        assert torch.all(augmented_x == x) & torch.all(augmented_y == y)

    def test_mixup(self):
        mixup = image_augmentations.MixUp(p=1.0)
        # Create dummy input and labels
        x = torch.cat(
            [torch.ones(16, 1, 100, 100), torch.zeros(16, 1, 100, 100)], dim=0
        )
        # Multiclass labels
        y = torch.cat([torch.ones(16, 2), torch.zeros(16, 2)], dim=0)
        # Apply CutMix augmentation
        augmented_x, augmented_y = mixup(x, y)
        # Assert the output shapes are correct
        assert augmented_x.shape == x.shape
        assert augmented_y.shape == y.shape

        # Because the images are all ones and zeros the mean of the pixels should be equal to the labels after being transformed
        assert torch.allclose(augmented_x.mean(dim=-1).mean(dim=-1), augmented_y)

        mixup = image_augmentations.MixUp(p=0)
        augmented_x, augmented_y = mixup(x, y)

        assert torch.all(augmented_x == x) & torch.all(augmented_y == y)
