import torch

from epochalyst.training.augmentation import utils


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


class TestUtils:
    def test_no_op(self):
        no_op = utils.NoOp()
        x = torch.rand(4, 1, 100, 100)
        augmented_x = no_op(x)

        assert torch.all(augmented_x == x)

    def test_custom_sequential(self):
        class DummyXStep:
            def __call__(self, x: torch.Tensor):
                return x + 1

        class DummyXYStep:
            def __call__(self, x: torch.Tensor, y: torch.Tensor):
                return x + 1, y + 1

        step1 = DummyXStep()
        step2 = DummyXYStep()

        sequential = utils.CustomSequential(x_transforms=[step1], xy_transforms=[step2])

        x = torch.ones(32, 1, 100)
        y = torch.zeros(32, 1)
        augmented_x, augmented_y = sequential(x, y)

        assert torch.all(augmented_x == x + 2)
        assert torch.all(augmented_y == y + 1)

    def test_custom_apply_one(self):
        class DummyXStep:
            def __init__(self, p):
                self.p = p

            def __call__(self, x: torch.Tensor):
                return x + 1

        class DummyXYStep:
            def __init__(self, p):
                self.p = p

            def __call__(self, x: torch.Tensor, y: torch.Tensor):
                return x, y + 1

        set_torch_seed(42)
        step1 = DummyXStep(p=0.33)
        step2 = DummyXStep(p=0.33)
        step3 = DummyXYStep(p=0.33)

        apply_one = utils.CustomApplyOne(x_transforms=[step1, step2])

        x = torch.ones(32, 1, 1)
        y = torch.zeros(32, 1)
        augmented_x, augmented_y = apply_one(x, y)

        assert torch.all(augmented_x == x + 1)

        apply_one = utils.CustomApplyOne(
            x_transforms=[step1, step2], xy_transforms=[step3]
        )
        augmented_x = x
        augmented_y = y
        for _ in range(10000):
            augmented_x, augmented_y = apply_one(augmented_x, augmented_y)
        # Assert that the xy transform is applied roughly 1/3 of the time
        assert torch.all(3300 <= augmented_y) & torch.all(augmented_y <= 3366)
        # Assert that the x transform is applied roughly 2/3 of the time
        assert torch.all(6633 <= augmented_x) & torch.all(augmented_x <= 6700)

    def test_audiomentations_compose(self):
        compose = utils.get_audiomentations().Compose([])
        transformer = utils.AudiomentationsCompose(compose=compose)
        x = torch.ones(32, 1, 100)
        augmented_x = transformer(x)

        assert torch.all(augmented_x == x)

    def test_audiomentations_compose_repr(self):
        compose = utils.get_audiomentations().Compose([])
        transformer = utils.AudiomentationsCompose(compose=compose)
        repr_str = repr(transformer)

        assert repr_str == ""

