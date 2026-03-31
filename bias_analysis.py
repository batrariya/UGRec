import numpy as np
import matplotlib.pyplot as plt

from evaluate import compute_exposure_from_model


############################################
# Compute Exposure Difference (Bias Magnitude)
############################################

def exposure_difference(male_exposure, female_exposure):

    genres = sorted(list(set(male_exposure.keys()) | set(female_exposure.keys())))

    male_total = sum(male_exposure.values())
    female_total = sum(female_exposure.values())

    print("\n===== Exposure Difference =====")

    total_diff = 0

    for g in genres:

        male_ratio = male_exposure[g] / male_total if male_total > 0 else 0
        female_ratio = female_exposure[g] / female_total if female_total > 0 else 0

        diff = abs(male_ratio - female_ratio)
        total_diff += diff

        print(
            f"{g:15s} | "
            f"Male: {male_ratio:.4f} | "
            f"Female: {female_ratio:.4f} | "
            f"Diff: {diff:.4f}"
        )

    print(f"\nTotal Bias Difference: {total_diff:.4f}")

    return total_diff


############################################
# Plot Bias Visualization
############################################

def plot_gender_bias(male_exposure, female_exposure, title="Gender Bias"):

    genres = sorted(list(set(male_exposure.keys()) | set(female_exposure.keys())))

    male_vals = [male_exposure[g] for g in genres]
    female_vals = [female_exposure[g] for g in genres]

    x = np.arange(len(genres))
    width = 0.35

    plt.figure(figsize=(12,6))

    plt.bar(x - width/2, male_vals, width, label='Male')
    plt.bar(x + width/2, female_vals, width, label='Female')

    plt.xlabel("Movie Genres")
    plt.ylabel("Recommendation Count")
    plt.title(title)

    plt.xticks(x, genres, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()


############################################
# Full Bias Analysis Pipeline
############################################

def run_bias_analysis(
    model,
    data,
    name="Model"
):

    print(f"\n===============================")
    print(f"Bias Analysis : {name}")
    print(f"===============================")

    # Use SAME exposure logic as evaluation.py
    # male_exp, female_exp = compute_exposure_from_model(
    #     model,
    #     adj,
    #     train_interactions,
    #     gender_dict,
    #     item_category
    # )

    male_exp, female_exp = compute_exposure_from_model(
        model,
        data
    )

    exposure_difference(male_exp, female_exp)

    plot_gender_bias(
        male_exp,
        female_exp,
        title=f"{name} Gender Bias"
    )