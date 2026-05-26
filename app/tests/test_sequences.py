from backend.core.config import DEFAULT_CONFIG
from backend.core.paths import MASTER_DATASET_PATH
from ml.utils.data_loading import load_master_dataset
from ml.utils.preprocessing import build_model_frame, chronological_train_test_split, fit_feature_scaler, transform_features
from ml.utils.sequences import make_sequences_from_block


def test_sequences_use_past_window_only():
    feature_cols = list(DEFAULT_CONFIG.var_feature_cols)
    df = load_master_dataset(MASTER_DATASET_PATH)
    model_df = build_model_frame(df, DEFAULT_CONFIG)
    split = chronological_train_test_split(model_df, DEFAULT_CONFIG)
    scaler = fit_feature_scaler(split.train_raw, feature_cols)
    train_scaled = transform_features(split.train_raw, scaler, feature_cols)

    X, y, dates = make_sequences_from_block(
        train_scaled,
        feature_cols,
        DEFAULT_CONFIG.target_col,
        DEFAULT_CONFIG.date_col,
        DEFAULT_CONFIG.seq_len,
    )

    assert X.shape[1] == DEFAULT_CONFIG.seq_len
    assert X.shape[2] == len(feature_cols)
    assert len(X) == len(y) == len(dates)
    assert dates[0] == train_scaled[DEFAULT_CONFIG.date_col].iloc[DEFAULT_CONFIG.seq_len - 1]
