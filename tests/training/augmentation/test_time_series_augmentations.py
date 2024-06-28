import numpy as np
import torch

from epochalyst.training.augmentation import time_series_augmentations


def set_torch_seed(seed: int = 42) -> None:
    """Set torch seed for reproducibility.

    :param seed: seed to set

    :return: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TestTimeSeriesAugmentations:
    def test_cutmix1d(self):
        set_torch_seed(42)
        cutmix1d = time_series_augmentations.CutMix1D(p=1.0)
        # Create dummy input and labels
        x = torch.cat([torch.ones(16, 1, 100), torch.zeros(16, 1, 100)], dim=0)
        # Multiclass labels
        y = torch.cat([torch.ones(16, 2), torch.zeros(16, 2)], dim=0)
        # Apply CutMix augmentation
        augmented_x, augmented_y = cutmix1d(x, y)

        # Assert the output shapes are correct
        assert augmented_x.shape == x.shape
        assert augmented_y.shape == y.shape

        # Because the images are all ones and zeros the mean of the pixels should be equal to the labels after being transformed
        assert torch.allclose(augmented_x.mean(dim=-1), augmented_y)

        cutmix1d = time_series_augmentations.CutMix1D(p=0)
        augmented_x, augmented_y = cutmix1d(x, y)
        assert torch.all(augmented_x == x) & torch.all(augmented_y == y)

    def test_mixup1d(self):
        set_torch_seed(42)
        mixup1d = time_series_augmentations.MixUp1D(p=1.0)
        # Create dummy input and labels
        x = torch.cat([torch.ones(16, 1, 100), torch.zeros(16, 1, 100)], dim=0)
        # Multiclass labels
        y = torch.cat([torch.ones(16, 2), torch.zeros(16, 2)], dim=0)
        # Apply CutMix augmentation
        augmented_x, augmented_y = mixup1d(x, y)

        # Assert the output shapes are correct
        assert augmented_x.shape == x.shape
        assert augmented_y.shape == y.shape

        # Because the images are all ones and zeros the mean of the pixels should be equal to the labels after being transformed
        assert torch.allclose(augmented_x.mean(dim=-1), augmented_y)

        mixup1d = time_series_augmentations.MixUp1D(p=0)
        augmented_x, augmented_y = mixup1d(x, y)
        assert torch.all(augmented_x == x) & torch.all(augmented_y == y)

    def test_mirror1d(self):
        set_torch_seed(42)
        mirror1d = time_series_augmentations.Mirror1D(p=1.0)
        x = torch.cat([torch.ones(32, 1, 50), torch.zeros(32, 1, 50)], dim=-1)

        augmented_x = mirror1d(x)

        # Assert the output shape is correct
        assert augmented_x.shape == x.shape

        # Assert x is mirrored
        assert torch.allclose(
            augmented_x,
            torch.cat([torch.zeros(32, 1, 50), torch.ones(32, 1, 50)], dim=-1),
        )

        mirror1d = time_series_augmentations.Mirror1D(p=0)
        augmented_x = mirror1d(x)
        assert torch.all(augmented_x == x)

    def test_random_amplitude_shift(self):
        set_torch_seed(42)
        low = 0.5
        high = 1.5
        random_amplitude_shift = time_series_augmentations.RandomAmplitudeShift(
            p=1.0, low=low, high=high
        )
        # Sum of 2 signals with the 2nd one being half the frequency of the first one
        x = torch.sin(torch.linspace(0, 2 * np.pi, 1000)) + torch.sin(
            torch.linspace(0, np.pi, 1000)
        )
        augmented_x = random_amplitude_shift(x)

        # Assert the output shape is correct
        assert augmented_x.shape == x.shape
        # Assert that the resulting signals amplitudes do not go over the bounds that have been set
        assert torch.all(
            torch.abs(torch.fft.rfft(x)) * low <= torch.abs(torch.fft.rfft(augmented_x))
        ) & torch.all(
            torch.abs(torch.fft.rfft(augmented_x))
            <= torch.abs(torch.fft.rfft(x)) * high
        )

        random_amplitude_shift = time_series_augmentations.RandomAmplitudeShift(p=0)
        augmented_x = random_amplitude_shift(x)
        assert torch.all(augmented_x == x)

    def test_random_phase_shift(self):
        set_torch_seed(42)
        random_phase_shift = time_series_augmentations.RandomPhaseShift(p=1.0)
        x = torch.sin(torch.linspace(0, 2 * np.pi, 1000))
        augmented_x = random_phase_shift(x)

        # Assert the output shape is correct
        assert augmented_x.shape == x.shape

        # Assert x is not equal to augmented x
        assert not torch.allclose(augmented_x, x)
        # Aseert that the absolute value of the rfft is still the same. Very high atol beacuse sin function isn't precise with 1000 points
        assert torch.allclose(
            torch.abs(torch.fft.rfft(x, dim=-1)),
            torch.abs(torch.fft.rfft(augmented_x, dim=-1)),
            atol=0.05,
        )
        # Assert that the mean is still around 0 and equal to the original mean
        assert torch.isclose(augmented_x.mean(), x.mean())
        assert torch.isclose(augmented_x.mean(), torch.tensor([0]).float())

        random_phase_shift = time_series_augmentations.RandomPhaseShift(p=0)
        augmented_x = random_phase_shift(x)
        assert torch.all(augmented_x == x)

    def test_reverse_1d(self):
        set_torch_seed(42)
        reverse1d = time_series_augmentations.Reverse1D(p=1.0)
        x = torch.sin(torch.linspace(0, 2 * np.pi, 1000)).unsqueeze(0)
        test_x = torch.sin(torch.linspace(np.pi, 3 * np.pi, 1000)).unsqueeze(0)
        augmented_x = reverse1d(x)

        # Assert the output shape is correct
        assert augmented_x.shape == x.shape
        # Assert the reversed sine wave is equal to 180 degrees phase shifted version
        assert torch.allclose(test_x, augmented_x, atol=0.0000005)

        reverse1d = time_series_augmentations.Reverse1D(p=0)
        augmented_x = reverse1d(x)
        assert torch.all(augmented_x == x)

    def test_subtract_channels(self):
        set_torch_seed(42)
        subtract_channels = time_series_augmentations.SubtractChannels(p=1.0)
        # Only works for multi-channel sequences
        x = torch.ones(32, 2, 100)
        augmented_x = subtract_channels(x)

        # Assert the output shape is correct
        assert augmented_x.shape == x.shape

        assert torch.allclose(torch.zeros(*augmented_x.shape), augmented_x)

        subtract_channels = time_series_augmentations.SubtractChannels(p=0)
        augmented_x = subtract_channels(x)
        assert torch.all(augmented_x == x)

    def test_find_window(self):
        # Create dummy donor and receiver signals
        energy_cutmix = time_series_augmentations.EnergyCutmix(p=1.0)
        donor = torch.tensor([[4, 1, 2, 6, 5]], dtype=torch.float32).unsqueeze(0)
        receiver = torch.tensor([[2, 3, 5, 1, 1]], dtype=torch.float32).unsqueeze(0)

        window_size = 2
        stride = 1
        donor_start, donor_end, receiver_start, receiver_end = energy_cutmix.find_window(donor, receiver, window_size, stride)

        # Check correct indices for maximum energy in donor and minimum in receiver
        assert donor_start == 3  
        assert donor_end == 5
        assert receiver_start == 3
        assert receiver_end == 5       

    def test_energy_cutmix(self):
        set_torch_seed(1)
        energycutmix = time_series_augmentations.EnergyCutmix(p=1.0)
        # Create dummy input and labels
        x = torch.cat([torch.ones(1, 1, 1000), torch.zeros(1, 1, 1000)], dim=0)
        # Multiclass labels
        y = torch.cat([torch.ones(1, 2), torch.zeros(1, 2)], dim=0)
        # Apply CutMix augmentation
        augmented_x, augmented_y = energycutmix(x, y)

        # Assert the output shapes are correct
        assert augmented_x.shape == x.shape
        assert augmented_y.shape == y.shape

        # first samples mean must be bound by the lower and upper bounds in the class
        assert energycutmix.low <= augmented_x[1].mean() <= energycutmix.high

        energycutmix = time_series_augmentations.EnergyCutmix(p=0)
        augmented_x, augmented_y = energycutmix(x, y)
        assert torch.all(augmented_x == x) & torch.all(augmented_y == y)

    def test_add_background_noise_wrapper(self):
        add_background_noise_wrapper = time_series_augmentations.AddBackgroundNoiseWrapper(p=1.0,
                                                                                           sounds_path='tests/training/augmentation/test_audio/white_noise.wav')
        x = torch.rand(44100, dtype=torch.float32)
        sr = 44100
        augmented_x = add_background_noise_wrapper(x, sr)

        # verify that not applying augmentation doesnt do anything
        assert not torch.all(x == augmented_x)
