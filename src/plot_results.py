import matplotlib.pyplot as plt
import numpy as np

def main():
    # Data
    models = ['Random (K=5)', 'Linear', 'LSTM']
    
    # Metrics from your provided data
    # Random: Precision 0.052, Recall 0.048, F1 0.042
    # Linear: Precision 0.202, Recall 0.178, F1 0.163
    # LSTM:   Precision 0.053, Recall 0.923, F1 0.097
    precision = [0.052, 0.202, 0.0532]
    recall = [0.048, 0.178, 0.9226]
    f1 = [0.042, 0.163, 0.0971]

    x = np.arange(len(models))  # the label locations
    width = 0.25  # the width of the bars

    # Create figure with high DPI for poster quality
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Colors: Dark Blue, Dark Orange/Gold, and Teal/Green
    color_precision = '#003366'  # Dark Navy Blue
    color_recall = '#E69F00'     # Colorblind-friendly Orange/Gold
    color_f1 = '#009E73'         # Colorblind-friendly Teal/Green

    # Plotting bars
    rects1 = ax.bar(x - width, precision, width, label='Precision', color=color_precision, edgecolor='black', linewidth=1.5)
    rects2 = ax.bar(x, recall, width, label='Recall', color=color_recall, edgecolor='black', linewidth=1.5)
    rects3 = ax.bar(x + width, f1, width, label='F1 Score', color=color_f1, edgecolor='black', linewidth=1.5)

    # Styling
    ax.set_ylabel('Score', fontsize=20, fontweight='bold', labelpad=15)
    ax.set_title('Performance Metrics by Model Architecture', fontsize=24, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=18, fontweight='bold')
    ax.tick_params(axis='y', labelsize=16)
    
    # Legend
    ax.legend(fontsize=16, frameon=True, fancybox=True, shadow=True, loc='upper left')

    # Grid
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # Adjust layout
    fig.tight_layout()

    # Save
    output_path = 'model_comparison_poster.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
