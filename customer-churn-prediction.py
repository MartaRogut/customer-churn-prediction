import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

#Uproszczone drzewo (rysunek)

def draw_simple_tree(save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    def node(x, y, text, fc="#f6f6f6"):
        ax.text(
            x, y, text,
            ha="center", va="center",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=fc, edgecolor="black"
            )
        )

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.03,
                    label, ha="center", fontsize=11)

    nodes = [
        (0.5, 0.85, "Problemy z płatnościami?", "#f6f6f6"),
        (0.25, 0.60, "Częste kontakty z supportem?", "#f6f6f6"),
        (0.75, 0.60, "Rzadkie korzystanie z usługi?", "#f6f6f6"),
        (0.15, 0.35, "ODEJDZIE", "#ffe5e5"),
        (0.35, 0.35, "MOŻE ODEJDZIE", "#fff2cc"),
        (0.65, 0.35, "MOŻE ODEJDZIE", "#fff2cc"),
        (0.85, 0.35, "ZOSTANIE", "#e6ffea"),
    ]

    for x, y, text, color in nodes:
        node(x, y, text, color)

    arrows = [
        (0.5, 0.80, 0.25, 0.65, "TAK"),
        (0.5, 0.80, 0.75, 0.65, "NIE"),
        (0.25, 0.55, 0.15, 0.40, "TAK"),
        (0.25, 0.55, 0.35, 0.40, "NIE"),
        (0.75, 0.55, 0.65, 0.40, "TAK"),
        (0.75, 0.55, 0.85, 0.40, "NIE"),
    ]

    for x1, y1, x2, y2, label in arrows:
        arrow(x1, y1, x2, y2, label)

    ax.set_title("Uproszczone drzewo decyzyjne", fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)

    plt.show()

#Wczytanie i przygotowanie danych
COLS = [
    "CustomerID", "Age", "Gender", "Tenure", "UsageFrequency", "SupportCalls",
    "PaymentDelay", "SubscriptionType", "ContractLength",
    "TotalSpend", "LastInteraction", "Churn"
]

# Churn to etykieta (target) -> NIE traktujemy jako cechy
NUM_COLS = [
    "Age", "Tenure", "UsageFrequency", "SupportCalls",
    "PaymentDelay", "TotalSpend", "LastInteraction"
]

CAT_COLS = ["Gender", "SubscriptionType", "ContractLength"]
def load_data(path: str) -> pd.DataFrame:

    df = pd.read_excel(path, header=None)
    s = df.iloc[:, 0].astype(str).str.strip()
    parts = s.str.split(",", n=len(COLS) - 1, expand=True)

    parts = parts.iloc[1:].copy()
    parts.columns = COLS

    for c in CAT_COLS:
        parts[c] = parts[c].astype(str).str.strip()

    for col in NUM_COLS:
        parts[col] = pd.to_numeric(parts[col], errors="coerce")

    # churn
    parts["Churn"] = pd.to_numeric(parts["Churn"], errors="coerce")

    # odrzuć wiersze bez churn
    parts = parts.dropna(subset=["Churn"])
    parts["Churn"] = parts["Churn"].astype(int)

    return parts


def ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)


def print_metrics(title: str, y_true, y_pred):
    print(f"\n{title}")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print("\nRaport:\n", classification_report(y_true, y_pred))
    print("Macierz pomyłek:\n", confusion_matrix(y_true, y_pred))


def save_or_show(fig_path: str | None):
    plt.tight_layout()
    if fig_path:
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=200)
    plt.show()

# MAIN
if __name__ == "__main__":
    ensure_dirs()

    draw_simple_tree(save_path="outputs/figures/simple_tree.png")
    train = load_data("training.xlsx")
    test = load_data("testing.xlsx")

    X_train = train.drop(["Churn", "CustomerID"], axis=1)
    y_train = train["Churn"].astype(int)

    X_test = test.drop(["Churn", "CustomerID"], axis=1)
    y_test = test["Churn"].astype(int)

    # one-hot na train+test razem (żeby te same kolumny po kodowaniu)
    full = pd.concat([X_train, X_test], axis=0)

    full_enc = pd.get_dummies(
        full,
        columns=CAT_COLS,
        drop_first=True
    )

    X_train_enc = full_enc.iloc[:len(X_train)]
    X_test_enc = full_enc.iloc[len(X_train):]

    # bazowe drzewo decyzyjne
    base_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    base_tree.fit(X_train_enc, y_train)
    y_pred = base_tree.predict(X_test_enc)

    print_metrics("DRZEWO BAZOWE", y_test, y_pred)

    plt.figure(figsize=(22, 10))
    plot_tree(
        base_tree,
        feature_names=X_train_enc.columns,
        class_names=["no churn", "churn"],
        filled=True,
        rounded=True,
        max_depth=4
    )
    plt.title("Bazowe drzewo decyzyjne")
    save_or_show("outputs/figures/base_tree.png")

    #tuning drzewa (GridSearchCV)
    param_grid = {
        "max_depth": [3, 4, 5],
        "min_samples_split": [10, 20, 40],
        "min_samples_leaf": [10, 20, 30]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        scoring="f1",
        cv=cv
    )

    grid.fit(X_train_enc, y_train)
    best_tree = grid.best_estimator_

    print("\nNAJLEPSZE PARAMETRY MODELU:")
    for param, value in grid.best_params_.items():
        print(f"  • {param:20s}: {value}")

    #ewaluacja najlepszego drzewa
    y_pred_best = best_tree.predict(X_test_enc)
    print_metrics("NAJLEPSZE DRZEWO", y_test, y_pred_best)

    plt.figure(figsize=(24, 12))
    plot_tree(
        best_tree,
        feature_names=X_train_enc.columns,
        class_names=["no churn", "churn"],
        filled=True,
        rounded=True,
        max_depth=5
    )
    plt.title("Najlepsze drzewo decyzyjne")
    save_or_show("outputs/figures/best_tree.png")

    #Random Forest (bazowy)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )

    rf.fit(X_train_enc, y_train)
    y_pred_rf = rf.predict(X_test_enc)

    print_metrics("BAZOWY RANDOM FOREST", y_test, y_pred_rf)

    importances = pd.Series(
        rf.feature_importances_,
        index=X_train_enc.columns
    ).sort_values(ascending=False)

    print("\nNajważniejsze cechy (Random Forest):")
    print(importances.head(10))

    #zapis feature importances do pliku
    importances.to_csv("outputs/feature_importances.csv", header=["importance"])
