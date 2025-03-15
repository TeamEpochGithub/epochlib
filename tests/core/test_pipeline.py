from epochlib.core import TrainingSystem, Pipeline
from epochlib.core import Transformer, TransformingSystem
import numpy as np


class TestPipeline:
    def test_pipeline_init(self):
        pipeline = Pipeline()
        assert pipeline is not None

    def test_pipeline_init_with_systems(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        label_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
            label_sys=label_system,
        )
        assert pipeline is not None

    def test_pipeline_train(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        label_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
            label_sys=label_system,
        )
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_y_system(self):
        x_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_x_system(self):
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_no_train_system(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        post_system = TransformingSystem()
        post_label_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=None,
            pred_sys=post_system,
            label_sys=post_label_system,
        )
        assert pipeline.train([1, 2], [1, 2]) == ([1, 2], [1, 2])

    def test_pipeline_train_no_refining_system(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        pipeline = Pipeline(x_sys=x_system, y_sys=y_system, train_sys=training_system)
        assert pipeline.train([1, 2, 3], [1, 2, 3]) == ([1, 2, 3], [1, 2, 3])

    def test_pipeline_train_1_x_transform_block(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        result = pipeline.train(np.array([1, 2, 3]), [1, 2, 3])
        assert np.array_equal(result[0], np.array([2, 4, 6])) and np.array_equal(
            result[1], np.array([1, 2, 3])
        )

    def test_pipeline_predict(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_predict_no_y_system(self):
        x_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_predict_no_systems(self):
        pipeline = Pipeline()
        assert pipeline.predict([1, 2, 3]) == [1, 2, 3]

    def test_pipeline_get_hash_no_change(self):
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        predicting_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=predicting_system,
        )
        assert x_system.get_hash() == ""

    def test_pipeline_get_hash_with_change(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem()
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert x_system.get_hash() != y_system.get_hash()
        assert pipeline.get_hash() != ""

    def test_pipeline_predict_system_hash(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem()
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem(steps=[transform1])
        pipeline = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert prediction_system.get_hash() != x_system.get_hash()
        assert pipeline.get_hash() != ""

    def test_pipeline_pre_post_hash(self):
        class TransformingBlock(Transformer):
            def transform(self, x):
                return x * 2

        transform1 = TransformingBlock()
        x_system = TransformingSystem(steps=[transform1])
        y_system = TransformingSystem()
        training_system = TrainingSystem()
        prediction_system = TransformingSystem(steps=[transform1])
        assert x_system.get_hash() == prediction_system.get_hash()
        pipeline1 = Pipeline(
            x_sys=x_system,
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        pipeline1_train_sys_hash = pipeline1.train_sys.get_hash()
        pipeline2 = Pipeline(
            x_sys=TransformingSystem(),
            y_sys=y_system,
            train_sys=training_system,
            pred_sys=prediction_system,
        )
        assert pipeline1_train_sys_hash != pipeline2.train_sys.get_hash()
