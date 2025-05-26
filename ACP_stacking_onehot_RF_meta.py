import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, f1_score, confusion_matrix, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier

# ====== Define Base Models (CNN, Transformer) ======
def create_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.05),  # # Added dropout
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.05),  # # Added dropout
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_transformer(input_shape, embed_dim=128, num_heads=4, ff_dim=128):
    inputs = Input(shape=input_shape)
    x = Conv1D(embed_dim, kernel_size=1, activation='relu')(inputs)

    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(0.05)(attn_output)  # Added dropout
    out1 = Add()([x, attn_output])
    out1 = LayerNormalization(epsilon=1e-6)(out1)

    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(embed_dim)(ffn)
    ffn = Dropout(0.05)(ffn)  # Added dropout
    out2 = Add()([out1, ffn])
    out2 = LayerNormalization(epsilon=1e-6)(out2)

    x = GlobalAveragePooling1D()(out2)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ====== Define Meta Model ======
def create_meta_model_rf(n_estimators=100, max_depth=7, random_state=42):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=random_state)

# ====== Evaluate Metrics ======
def evaluate_model(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_pred_prob),
        "pr_auc": average_precision_score(y_true, y_pred_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred)
    }

# ====== Run Stacking ======
def run_stacking(X_train, y_train, X_test, y_test, n_repeats=3):
    final_results = {metric: [] for metric in ["accuracy", "balanced_accuracy", "auc", "pr_auc", "mcc", "precision", "recall", "specificity", "f1"]}

    for repeat in range(n_repeats):
        print(f"\n===== Training Run {repeat+1}/{n_repeats} =====")

        models = [
            create_cnn(X_train.shape[1:]),
            create_transformer(X_train.shape[1:])
        ]

        model_preds = []
        for model in models:
            model.fit(X_train, y_train, validation_split=0.3, epochs=30, batch_size=32, verbose=0)
            model_preds.append(model.predict(X_test).flatten())

        meta_X = np.column_stack(model_preds)
        meta_model = create_meta_model_rf()
        meta_model.fit(meta_X, y_test)
        meta_y_pred_prob = meta_model.predict_proba(meta_X)[:, 1]

        metrics = evaluate_model(y_test, meta_y_pred_prob)

        for metric in final_results:
            final_results[metric].append(metrics[metric])

        print(" → " + ", ".join([f"{name}: {value:.3f}" for name, value in metrics.items()]))

    print("\n=== Final Evaluation (Mean ± Std) ===")
    for metric, values in final_results.items():
        print(f"{metric.capitalize():18}: {np.mean(values):.3f} ± {np.std(values):.3f}")

# ====== Main Execution ======
def main():
    X_train = pd.read_csv("3.ACP-DL_acp740_x_train_onehot_esm.csv").values
    y_train = pd.read_csv("3.ACP-DL_acp740_y_train.csv",index_col=0).values.ravel()
    X_test = pd.read_csv("3.ACP-DL_acp740_x_test_onehot_esm.csv").values
    y_test = pd.read_csv("3.ACP-DL_acp740_y_test.csv",index_col=0).values.ravel()

    max_length = X_train.shape[1] // 20

    X_train = X_train.reshape((-1, max_length, 20))
    X_test = X_test.reshape((-1, max_length, 20))

    print("✅ Input shapes:")
    print("  X_train:", X_train.shape)
    print("  X_test :", X_test.shape)

    run_stacking(X_train, y_train, X_test, y_test, n_repeats=3)

if __name__ == "__main__":
    main()