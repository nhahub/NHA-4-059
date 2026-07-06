from src.utils.experiment_config import ExperimentConfig, set_seed
from src.utils.experiment_log import ExperimentLogger


def test_experiment_config_round_trips_through_json(tmp_path):
    """FR-8: experiment configs must save and reload identically."""
    config = ExperimentConfig(
        classifier_path="clf.pkl",
        examples_path="examples.pkl",
        methods=["clean", "blur"],
        output_dir="outputs",
        seed=7,
        notes="unit test run",
    )
    config_path = tmp_path / "config.json"
    config.save(str(config_path))

    loaded = ExperimentConfig.load(str(config_path))

    assert loaded == config


def test_set_seed_does_not_raise():
    set_seed(123)  # only asserting this runs cleanly with/without torch installed


def test_experiment_logger_appends_rows_with_header_once(tmp_path):
    log_path = tmp_path / "experiment_log.csv"
    logger = ExperimentLogger(log_path)

    exp_id_1 = logger.log(
        model_name="CLIP-RN50",
        dataset_id="imagenet_trucks",
        xai_method="gradcam",
        accuracy=0.9,
        focus_score=0.4,
        shortcut_detected=True,
        heatmap_path="outputs/heatmaps",
    )
    exp_id_2 = logger.log(
        model_name="CLIP-RN50",
        dataset_id="imagenet_trucks",
        xai_method="gradcam",
        accuracy=0.91,
        focus_score=0.2,
        shortcut_detected=False,
        heatmap_path="outputs/heatmaps",
    )

    assert exp_id_1 != exp_id_2

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert lines[0].startswith("exp_id,")
