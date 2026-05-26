from backend.core.config import DEFAULT_CONFIG
from backend.core.paths import MASTER_DATASET_PATH
from ml.utils.data_loading import load_master_dataset
from ml.utils.preprocessing import build_model_frame, chronological_train_test_split


def test_fixed_learning_window_and_variable_test():
    df = load_master_dataset(MASTER_DATASET_PATH)
    model_df = build_model_frame(df, DEFAULT_CONFIG)
    split = chronological_train_test_split(model_df, DEFAULT_CONFIG)
    expected_test_window = max(
        1,
        int(round(DEFAULT_CONFIG.train_window * DEFAULT_CONFIG.test_window_ratio)),
    )

    assert len(split.train_full_raw) == DEFAULT_CONFIG.train_window
    assert len(split.test_raw) == expected_test_window
    assert split.train_full_raw["date"].max() < split.test_raw["date"].min()
