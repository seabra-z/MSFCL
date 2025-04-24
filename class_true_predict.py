import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import matplotlib
from matplotlib.font_manager import FontProperties
import os


try:
    chinese_fonts = [f for f in matplotlib.font_manager.findSystemFonts()
                     if ('simhei' in f.lower() or 'simsun' in f.lower() or
                         'microsoft yahei' in f.lower() or 'wqy' in f.lower() or
                         'noto sans cjk' in f.lower())]

    if chinese_fonts:
        plt.rcParams['font.family'] = FontProperties(fname=chinese_fonts[0]).get_name()
    else:
        use_english = True
except:
    use_english = True


plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300  
plt.rcParams['figure.dpi'] = 100  


output_dir = "ddi_analysis_results"
os.makedirs(output_dir, exist_ok=True)


def load_and_process_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            pred, true, ddi = line.strip().split()
            ddi_types = [int(d) for d in str(ddi)]
            data.append([int(pred), int(true), ddi_types])
    return pd.DataFrame(data, columns=['pred', 'true', 'ddi_types'])


def plot_confusion_matrices(methods_data):
    """绘制所有方法的混淆矩阵对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    risk_labels = ['Major', 'Moderate', 'Minor']

    for idx, (method_name, df) in enumerate(methods_data.items()):
        conf_matrix = pd.crosstab(df['true'], df['pred'], normalize='index') * 100

        sns.heatmap(conf_matrix, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=risk_labels, yticklabels=risk_labels, ax=axes[idx])
        axes[idx].set_title(method_name, fontsize=14)
        axes[idx].set_xlabel('Predicted Label', fontsize=12)
        axes[idx].set_ylabel('True Label', fontsize=12)

    plt.suptitle('Confusion Matrices Comparison', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrices.png", bbox_inches='tight')
    plt.close()


def analyze_ddi_accuracy(df):
    """分析每种DDI类型的预测准确率"""
    ddi_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})

    for _, row in df.iterrows():
        for ddi_type in row['ddi_types']:
            ddi_accuracy[ddi_type]['total'] += 1
            if row['pred'] == row['true']:
                ddi_accuracy[ddi_type]['correct'] += 1

    return {k: v['correct'] / v['total'] * 100 for k, v in ddi_accuracy.items()}


def plot_ddi_accuracy_comparison(methods_data):
    """绘制不同方法在各DDI类型上的准确率对比"""
    ddi_accuracies = {
        method: analyze_ddi_accuracy(df)
        for method, df in methods_data.items()
    }

    ddi_types = list(range(1, 9))
    ddi_names = [
        "Digestive/Metabolic", "Blood/Hematopoietic", "Dermatological",
        "Systemic Hormones", "Antineoplastic/Immunomodulating", "Antiparasitic",
        "Respiratory", "Various"
    ]

    fig, ax = plt.subplots(figsize=(14, 9))
    x = np.arange(len(ddi_types))
    width = 0.25

    for i, (method, accuracies) in enumerate(ddi_accuracies.items()):
        acc_values = [accuracies.get(t, 0) for t in ddi_types]
        ax.bar(x + i * width, acc_values, width, label=method)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Prediction Accuracy by DDI Type', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ddi_names, rotation=45, ha='right')
    ax.legend(fontsize=12,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.18), 
              ncol=3,  
              columnspacing=1.5,  
              frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylim(80, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/ddi_accuracy_comparison.png", bbox_inches='tight')
    plt.close()


def analyze_ddi_combinations(df, filter_correct=False):
    """分析DDI组合的预测模式"""
    combination_patterns = defaultdict(lambda: {'correct': 0, 'total': 0})

    for _, row in df.iterrows():
        ddi_combo = tuple(sorted(row['ddi_types']))
        combination_patterns[ddi_combo]['total'] += 1
        if row['pred'] == row['true']:
            combination_patterns[ddi_combo]['correct'] += 1
    if filter_correct:
        return {k: v['correct'] for k, v in combination_patterns.items()}

    return combination_patterns


def plot_top_combinations(methods_data):
    """绘制最常见DDI组合的预测准确率"""
    all_combinations = defaultdict(int)
    for df in methods_data.values():
        patterns = analyze_ddi_combinations(df)
        for combo, stats in patterns.items():
            all_combinations[combo] += stats['total']

    top_combos = sorted(all_combinations.items(), key=lambda x: x[1], reverse=True)[:10]

    combo_labels = ['+'.join(map(str, combo)) for combo, _ in top_combos]
    method_accuracies = []

    for method, df in methods_data.items():
        patterns = analyze_ddi_combinations(df)
        accuracies = []
        for combo, _ in top_combos:
            if combo in patterns and patterns[combo]['total'] > 0:
                acc = (patterns[combo]['correct'] / patterns[combo]['total']) * 100
            else:
                acc = 0
            accuracies.append(acc)
        method_accuracies.append(accuracies)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(combo_labels))
    width = 0.25

    for i, (method, accuracies) in enumerate(zip(methods_data.keys(), method_accuracies)):
        ax.bar(x + i * width, accuracies, width, label=method)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Prediction Accuracy for Top DDI Combinations', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(combo_labels, rotation=45, ha='right')
    ax.legend(fontsize=12,
              loc='upper center',
              bbox_to_anchor=(0.5, 1.18),
              ncol=3,
              columnspacing=1.5,
              frameon=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    ax.set_ylim(80, 100)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_combinations_accuracy.png", bbox_inches='tight')
    plt.close()


def analyze_error_patterns(methods_data):
    """分析错误预测模式（合并到单图）"""
    plt.figure(figsize=(14, 9))

    ddi_names = [
        "Digestive/Metabolic", "Blood/Hematopoietic", "Dermatological",
        "Systemic Hormones", "Antineoplastic/Immunomodulating", "Antiparasitic",
        "Respiratory", "Various"
    ]

    error_data = {}
    for method_name, df in methods_data.items():
        error_df = df[df['pred'] != df['true']].copy()
        ddi_error_counts = defaultdict(int)
        for _, row in error_df.iterrows():
            for ddi_type in row['ddi_types']:
                ddi_error_counts[ddi_type] += 1
        error_data[method_name] = [ddi_error_counts.get(t + 1, 0) for t in range(8)]

    x = np.arange(len(ddi_names))
    width = 0.25  
    for i, (method, counts) in enumerate(error_data.items()):
        offset = width * i  
        plt.bar(x + offset, counts, width, label=method)

    plt.ylabel('Error Count', fontsize=12)
    plt.title('Error Distribution Comparison by DDI Type', fontsize=14)
    plt.xticks(x + width, ddi_names, rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.legend(fontsize=12,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.18),
               ncol=3,
               columnspacing=1.5,
               frameon=True)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution_combined.png", bbox_inches='tight')
    plt.close()

def analyze_risk_level_performance(methods_data):
    """分析不同风险等级的预测性能"""
    risk_levels = [0, 1, 2]  # Major, Moderate, Minor
    risk_names = ['Major', 'Moderate', 'Minor']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, risk in enumerate(risk_levels):
        accuracies = []
        for method, df in methods_data.items():
            risk_df = df[df['true'] == risk]
            accuracy = (risk_df['pred'] == risk_df['true']).mean() * 100
            accuracies.append(accuracy)

        axes[i].bar(list(methods_data.keys()), accuracies, color=['royalblue', 'darkorange', 'forestgreen'])
        axes[i].set_title(f"{risk_names[i]} Risk Level Accuracy", fontsize=12)
        axes[i].set_ylabel('Accuracy (%)', fontsize=10)
        axes[i].set_ylim(80, 100)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

        for j, acc in enumerate(accuracies):
            axes[i].text(j, acc + 0.5, f'{acc:.1f}%', ha='center', fontsize=9)

    plt.suptitle('Performance Comparison by Risk Level', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/risk_level_performance.png", bbox_inches='tight')
    plt.close()


def analyze_misclassification_patterns(methods_data):
    """分析错误分类模式"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    risk_labels = ['Major', 'Moderate', 'Minor']

    for idx, (method_name, df) in enumerate(methods_data.items()):
        error_matrix = np.zeros((3, 3))

        for _, row in df.iterrows():
            if row['pred'] != row['true']:
                error_matrix[row['true'], row['pred']] += 1

        for i in range(3):
            error_matrix[i, i] = 0

        sns.heatmap(error_matrix, annot=True, fmt='g', cmap='Reds',
                    xticklabels=risk_labels, yticklabels=risk_labels, ax=axes[idx])
        axes[idx].set_title(f"{method_name} Misclassification Patterns", fontsize=12)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)

    plt.suptitle('Misclassification Pattern Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/misclassification_patterns.png", bbox_inches='tight')
    plt.close()


def plot_correct_distribution_comparison(methods_data):
    """各风险等级正确预测的DDI类型分布对比"""
    risk_labels = ['Major', 'Moderate', 'Minor']
    ddi_names = [
        "Digestive/Metabolic", "Blood/Hematopoietic", "Dermatological",
        "Systemic Hormones", "Antineoplastic", "Antiparasitic",
        "Respiratory", "Various"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    plt.suptitle("Correct Prediction DDI Type Distribution Comparison", y=1.05)

    for risk_level in range(3):
        risk_data = []
        for method, df in methods_data.items():
            correct_df = df[(df['pred'] == df['true']) & (df['true'] == risk_level)]
            ddi_counts = defaultdict(int)
            for _, row in correct_df.iterrows():
                for ddi_type in row['ddi_types']:
                    ddi_counts[ddi_type] += 1
            risk_data.append(ddi_counts)

        plot_data = []
        for method_idx, method_counts in enumerate(risk_data):
            total = sum(method_counts.values()) or 1
            percentages = [(method_counts.get(t, 0) / total) * 100 for t in range(1, 9)]
            plot_data.append(percentages)

        x = np.arange(8)
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for method_idx, (method, percentages) in enumerate(zip(methods_data.keys(), plot_data)):
            axes[risk_level].bar(x + method_idx * width, percentages, width,
                                 label=method, color=colors[method_idx])

        axes[risk_level].set_xticks(x + width)
        axes[risk_level].set_xticklabels(ddi_names, rotation=45, ha='right')
        axes[risk_level].set_title(f"{risk_labels[risk_level]} Risk Level")
        axes[risk_level].set_ylabel("Percentage (%)")
        axes[risk_level].grid(axis='y', linestyle='--', alpha=0.7)
        axes[risk_level].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/correct_distribution_comparison.png", bbox_inches='tight')
    plt.close()


def plot_error_distribution_comparison(methods_data, top_errors=5):
    """错误预测模式对比分析"""
    error_labels = ['Major', 'Moderate', 'Minor']
    ddi_names = [
        "Digestive/Metabolic", "Blood/Hematopoietic", "Dermatological",
        "Systemic Hormones", "Antineoplastic", "Antiparasitic",
        "Respiratory", "Various"
    ]

    all_error_patterns = defaultdict(int)
    for method, df in methods_data.items():
        error_df = df[df['pred'] != df['true']]
        for _, row in error_df.iterrows():
            pattern = (row['true'], row['pred'])
            all_error_patterns[pattern] += 1

    common_errors = sorted(all_error_patterns.items(),
                           key=lambda x: x[1], reverse=True)[:top_errors]

    for error_idx, ((true_l, pred_l), _) in enumerate(common_errors):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle(f"Error Pattern: {error_labels[true_l]} → {error_labels[pred_l]}", y=1.05)

        for method_idx, (method, df) in enumerate(methods_data.items()):
            error_df = df[(df['true'] == true_l) & (df['pred'] == pred_l)]
            ddi_counts = defaultdict(int)
            for _, row in error_df.iterrows():
                for ddi_type in row['ddi_types']:
                    ddi_counts[ddi_type] += 1
            sorted_ddi = sorted(ddi_counts.items(), key=lambda x: x[1], reverse=True)
            types = [t for t, _ in sorted_ddi]
            counts = [c for _, c in sorted_ddi]

            axes[method_idx].bar([ddi_names[t - 1] for t in types[:5]], counts[:5],
                                 color='indianred')
            axes[method_idx].set_title(f"{method}")
            axes[method_idx].set_ylabel("Count")
            axes[method_idx].tick_params(axis='x', rotation=45)
            axes[method_idx].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_comparison_{error_labels[true_l]}_to_{error_labels[pred_l]}.png",
                    bbox_inches='tight')
        plt.close()

def plot_high_frequency_combinations(methods_data, top_n=5, mode='correct'):
    """分析并绘制前N个高频正确/错误组合"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    plt.suptitle(f'Top {top_n} {mode.capitalize()} Combinations Comparison', y=0.95)

    for method_idx, (method, df) in enumerate(methods_data.items()):
        if mode == 'correct':
            combo_data = analyze_ddi_combinations(df, filter_correct=True)
        else:
            error_df = df[df['pred'] != df['true']]
            combo_data = analyze_ddi_combinations(error_df, filter_correct=False)
            combo_data = {k: v['total'] for k, v in combo_data.items()}

        top_combos = sorted(combo_data.items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels = ['+'.join(map(str, k)) for k, v in top_combos]
        values = [v for k, v in top_combos]

        axes[method_idx].barh(labels[::-1], values[::-1], color='steelblue' if mode == 'correct' else 'indianred')
        axes[method_idx].set_title(f'{method}')
        axes[method_idx].set_xlabel('Count')
        axes[method_idx].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/top_{top_n}_{mode}_combinations.png", bbox_inches='tight')
    plt.close()


# 主程序
def main():
    methods = {
        'MSFCL': 'MSFCL/true_pred_label_addreadout3.txt',
        'MRGCDDI': 'MRGCDDI/true_pred_label_addreadout3.txt',
        'MRCGNN': 'MRCGNN-main/codes for MRCGNN/true_pred_label_addreadout3.txt'
    }

    methods_data = {
        method: load_and_process_data(path)
        for method, path in methods.items()
    }

    plot_confusion_matrices(methods_data)
    plot_ddi_accuracy_comparison(methods_data)
    plot_top_combinations(methods_data)
    analyze_error_patterns(methods_data)
    analyze_risk_level_performance(methods_data)
    analyze_misclassification_patterns(methods_data)

    plot_correct_distribution_comparison(methods_data)
    plot_error_distribution_comparison(methods_data)
    plot_high_frequency_combinations(methods_data, mode='correct')
    plot_high_frequency_combinations(methods_data, mode='error')

    stats_results = statistical_analysis(methods_data)
    print(stats_results)

    print(f"\n所有分析结果已保存到 {output_dir} 目录")


if __name__ == "__main__":
    main()
