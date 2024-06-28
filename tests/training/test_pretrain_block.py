import pytest

from epochalyst.training import PretrainBlock


class TestPretrainBlock:
    def test_pretrain_block_init(self):
        pb = PretrainBlock()
        assert pb is not None

    def test_pretrain_block_train(self):
        with pytest.raises(NotImplementedError):
            pb = PretrainBlock()
            pb.train(None, None, None)

    def test_pretrain_block_predict(self):
        with pytest.raises(NotImplementedError):
            pb = PretrainBlock()
            pb.predict(None)

    def test_pretrain_block_train_split_hash(self):
        pb = PretrainBlock()
        initial_hash = pb.get_hash()
        pb.train_split_hash([1, 2, 3])
        assert pb.get_hash() != initial_hash
