# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:08:08 2024
Collection of metrics for bias assessment in tabular datasets.
@author: Vasileios C. Pezoulas
"""

import pandas as pd
import numpy as np
import os
import argparse
import warnings
import matplotlib.pyplot as plt
import matplotlib
import json
from minisom import MiniSom
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.stats import chi2_contingency


matplotlib.use('Agg')
warnings.filterwarnings('always')
# os.chdir(os.getcwd()) # what is the point?


def map_categorical_to_integers(df):
    mappings = {}
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].astype('category')
        mappings[column] = dict(enumerate(df[column].cat.categories))
        df[column] = df[column].cat.codes
    return df, mappings


def group_ages(df):
    age_mapping = {}
    if 'age' in df.columns:
        age_bins = [0, 39, 59, float('inf')]
        age_labels = [1, 2, 3]
        df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
        age_mapping = {1: '20-39', 2: '40-59', 3: '60+'}
    return df, age_mapping


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))


def plot_bias_metrics(metrics, file_path, cluster_id=None):
    thresholds = {
        'Class Imbalance (CI)': 0.1,
        'Difference in Proportion Labels (DPL)': 0.1,
        'Demographic Disparity (DD)': 0.1,
        'Jensen-Shannon Divergence (JS)': 0.1,
        'L2 Norm': 0.1,
        'KS value': 0.1,
        'NMI': 0.1,
        'BR': (0.8, 1.25),
        'BD': 0.1,
        'CORR': 0.1,
        'TVD': 0.1,
        'Conditional Demographic Disparity (CDD)': 0.1,
        'Conditional Mutual Information (NCMI)': 0.1,
        'Conditional Binary Difference (CBD)': 0.1,
        'Logistic Regression Coefficient (LR)': 0.1,
        'Logistic Regression Intercept (Intercept)': (-2, 2)
    }

    METRIC_LABELS_ORDER = [
        'Class Imbalance (CI)',
        'Difference in Proportion Labels (DPL)',
        'Demographic Disparity (DD)',
        'Jensen-Shannon Divergence (JS)',
        'L2 Norm',
        'KS value',
        'Normalized Mutual Information (NMI)',
        'Binary Ratio (BR)',
        'Binary Difference (BD)',
        'Pearson Correlation (CORR)',
        'Total Variation Distance (TVD)',
        'Conditional Demographic Disparity (CDD)',
        'Conditional Mutual Information (NCMI)',
        'Conditional Binary Difference (CBD)',
        'Logistic Regression Coefficient (LR)']

    METRIC_TITLES_ORDER = [
        'Class Imbalance (CI)',
        'Difference in Proportion Labels (DPL)',
        'Demographic Disparity (DD)',
        'Jensen-Shannon Divergence (JS)',
        'L2 Norm',
        'KS value',
        'Normalized Mutual Information (NMI)',
        'Binary Ratio (BR)',
        'Binary Difference (BD)',
        'Pearson Correlation (CORR)',
        'Total Variation Distance (TVD)',
        'Conditional Demographic Disparity (CDD)',
        'Conditional Mutual Information (NCMI)',
        'Conditional Binary Difference (CBD)',
        'Logistic Regression Coefficient (LR)']

    metric_values = [metrics[label] for label in METRIC_LABELS_ORDER if label in metrics]
    metric_labels = [label for label in METRIC_LABELS_ORDER if label in metrics]
    metric_titles = [title for title in METRIC_TITLES_ORDER if title in metrics]

    num_metrics = len(metric_values)
    num_rows = (num_metrics + 4) // 5

    plt.figure(figsize=(15, 3 * num_rows))
    light_red = '#f08080'
    light_green = '#8fbc8f'

    for i in range(num_metrics):
        plt.subplot(num_rows, 5, i + 1)
        if metric_values[i] is not None:
            bars = plt.barh(['original'], [metric_values[i]], color=['grey'], height=0.5)
            for bar in bars:
                xval = bar.get_width()
                plt.text(xval - 0.05 if xval < 0 else xval + 0.05, bar.get_y() + bar.get_height() / 2, round(xval, 2),
                         ha='left' if xval < 0 else 'right', va='center', fontsize=10, color='black')

        plt.axvline(0, color='black', linewidth=0.8)
        plt.xlabel(metric_labels[i], fontsize=10)
        plt.title(metric_titles[i], fontsize=12)

        threshold = thresholds.get(metric_labels[i], 0.1)
        if metric_labels[i] == 'BR':
            plt.xlim(-1, 1.25)
            plt.axvspan(-1, 0.8, facecolor=light_red, alpha=0.5, label='Bias')
            plt.axvspan(0.8, 1.25, facecolor=light_green, alpha=0.5, label='Fair')
            plt.axvspan(1.25, 2, facecolor=light_red, alpha=0.5)
        elif metric_labels[i] == 'Logistic Regression Coefficient (LR)' or metric_labels[i] == 'Logistic Regression Intercept (Intercept)':
            plt.xlim(-2.5, 2.5)
            plt.axvspan(-2.5, -2, facecolor=light_red, alpha=0.5, label='Bias')
            plt.axvspan(-2, 2, facecolor=light_green, alpha=0.5, label='Fair')
            plt.axvspan(2, 2.5, facecolor=light_red, alpha=0.5)
        else:
            plt.xlim(-1, 1)
            plt.axvspan(-1, -threshold, facecolor=light_red, alpha=0.5, label='Bias')
            plt.axvspan(-threshold, threshold, facecolor=light_green, alpha=0.5, label='Fair')
            plt.axvspan(threshold, 1, facecolor=light_red, alpha=0.5)

        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.yticks([])

        if i == 0:
            plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    if cluster_id is not None:
        plot_file_path = os.path.splitext(file_path)[0] + f'_cluster_{cluster_id}_metrics.png'
    else:
        plot_file_path = os.path.splitext(file_path)[0] + '_metrics.png'
    plt.savefig(plot_file_path, dpi=400)
    plt.show()
    plt.close()


def plot_silhouette_scores(cluster_counts, silhouette_scores, optimal_clusters=None):
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, silhouette_scores, marker='o', linestyle='-', color='b')
    
    if optimal_clusters is not None:
        optimal_index = cluster_counts.index(optimal_clusters)
        optimal_score = silhouette_scores[optimal_index]
        plt.scatter(optimal_clusters, optimal_score, color='red', s=100, label=f'Optimal ({optimal_clusters} clusters)')
        plt.legend()
    
    plt.title('Silhouette scores by Number of Clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.grid(True)
    plt.xticks(cluster_counts)
    plt.show()

    
def plot_db_scores(cluster_counts, db_scores, optimal_clusters=None):
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_counts, db_scores, marker='o', linestyle='-', color='b')
    
    if optimal_clusters is not None:
        optimal_index = cluster_counts.index(optimal_clusters)
        optimal_score = db_scores[optimal_index]
        plt.scatter(optimal_clusters, optimal_score, color='red', s=100, label=f'Optimal ({optimal_clusters} clusters)')
        plt.legend()
    
    plt.title('Davies-Bouldin scores by number of clusters')
    plt.xlabel('Number of clusters')
    plt.ylabel('Davies-Bouldin score')
    plt.grid(True)
    plt.xticks(cluster_counts)
    plt.show()


def perform_clustering(data, num_clusters):
    som_dim_x = int(np.ceil(np.sqrt(num_clusters)))
    som_dim_y = int(np.ceil(num_clusters / som_dim_x))
    sigma_value = min(som_dim_x, som_dim_y) / 2

    # print(f"Performing clustering with {som_dim_x}x{som_dim_y} grid and sigma={sigma_value}")

    som = MiniSom(som_dim_x, som_dim_y, data.shape[1], sigma=sigma_value, learning_rate=0.5)
    som.random_weights_init(data.values)
    som.train_random(data.values, 1000)

    labels = np.array([som.winner(d)[0] * som_dim_y + som.winner(d)[1] for d in data.values])
    
    max_label = som_dim_x * som_dim_y - 1
    labels = np.clip(labels, 0, max_label)
    
    cluster_map = {i: [] for i in range(max_label + 1)}
    
    for idx, label in enumerate(labels):
        cluster_map[label].append(idx)

    return labels, cluster_map, som


def find_optimal_clusters_db(data, max_clusters):
    db_scores = []
    cluster_counts = []
    for clusters in range(2, max_clusters + 1):
        labels, cluster_map, _ = perform_clustering(data, clusters)
        unique_labels = np.unique(labels)
        
        non_empty_labels = [label for label in unique_labels if len(cluster_map[label]) > 0]
        if len(non_empty_labels) > 1:
            filtered_labels = np.array([label if label in non_empty_labels else -1 for label in labels])
            filtered_data = data[filtered_labels != -1]
            score = davies_bouldin_score(filtered_data.values, filtered_labels[filtered_labels != -1])
            db_scores.append(score)
            cluster_counts.append(len(non_empty_labels))
            # print(f"DB Score for {clusters} clusters: {score} (Effective Clusters: {len(non_empty_labels)})")
            print(f"DB Score for {clusters} clusters: {score}")
        else:
            print(f"Insufficient clusters ({len(non_empty_labels)}) for DB score calculation at {clusters} clusters.")
    
    min_score_index = db_scores.index(min(db_scores))
    optimal_clusters = cluster_counts[min_score_index]

    return cluster_counts, db_scores, optimal_clusters


def find_optimal_clusters_silh(data, max_clusters):
    silhouette_scores = []
    cluster_counts = list(range(2, max_clusters + 1))
    
    for clusters in cluster_counts:
        labels, cluster_map, _ = perform_clustering(data, clusters)
        unique_labels = np.unique(labels)
        non_empty_labels = [label for label in unique_labels if len(cluster_map[label]) > 0]
        if len(non_empty_labels) > 1:
            filtered_labels = np.array([label if label in non_empty_labels else -1 for label in labels])
            filtered_data = data[filtered_labels != -1]
            score = silhouette_score(filtered_data.values, filtered_labels[filtered_labels != -1])
            silhouette_scores.append(score)
            # print(f"Silhouette Score for {clusters} clusters: {score} (Effective Clusters: {len(non_empty_labels)})")
            print(f"Silhouette Score for {clusters} clusters: {score}")
        else:
            silhouette_scores.append(-1)
            print(f"Insufficient clusters ({len(non_empty_labels)}) for Silhouette score calculation at {clusters} clusters.")
    
    max_score_index = silhouette_scores.index(max(silhouette_scores))
    optimal_clusters = cluster_counts[max_score_index]

    return cluster_counts, silhouette_scores, optimal_clusters


def calculate_generalized_imbalance(df, facet_name):
    facet_counts = df[facet_name].value_counts()
    total = facet_counts.sum()
    proportions = facet_counts / total
    imbalance = proportions.max() - proportions.min()
    return imbalance


def calculate_difference_in_proportions(na1, na, nd1, nd):
    if na == 0 or nd == 0:
        raise ValueError("Total number of members for either facet a or d cannot be zero.")

    qa = na1 / na if na > 0 else 0
    qd = nd1 / nd if nd > 0 else 0

    dpl = qa - qd
    return dpl


def kullback_leibler_divergence(Pa, Pd):
    Pa = np.array(Pa)
    Pd = np.array(Pd)
    Pd = np.where(Pd == 0, 1e-10, Pd)
    nonzero_Pa = Pa > 0
    kl_divergence = np.sum(Pa[nonzero_Pa] * np.log(Pa[nonzero_Pa] / Pd[nonzero_Pa]))
    
    return kl_divergence


def generalized_demographic_disparity(df, facet_name, outcome_name, reference_group=None):
    group_proportions = df.groupby(facet_name)[outcome_name].value_counts(normalize=True).unstack(fill_value=0)
    
    if reference_group:
        reference_proportions = group_proportions.loc[reference_group]
    else:
        reference_proportions = df[outcome_name].value_counts(normalize=True)
    
    disparity = group_proportions - reference_proportions

    return disparity


def generalized_conditional_demographic_disparity(df, facet_name, outcome_name, subgroup_column):
    if subgroup_column not in df.columns:
        print(f"Subgroup column '{subgroup_column}' not found in the dataframe.")
        return None

    df = df.dropna(subset=[subgroup_column])
    
    unique_subgroups = df[subgroup_column].nunique()
    if unique_subgroups < 2:
        print(f"Not enough unique values in the subgroup column '{subgroup_column}'.")
        return None

    subgroup_proportions = df.groupby([subgroup_column, facet_name])[outcome_name].value_counts(normalize=True).unstack(fill_value=0)
    overall_proportions = df[outcome_name].value_counts(normalize=True).reindex(subgroup_proportions.columns, fill_value=0)
    
    print("Subgroup Proportions:")
    print(subgroup_proportions)
    print("Overall Proportions:")
    print(overall_proportions)
    
    subgroup_proportions = subgroup_proportions.reset_index()
    subgroup_proportions.columns = ['subgroup', 'facet', 'outcome_0', 'outcome_1']
    overall_proportions = overall_proportions.values
    
    aggregated_disparity = pd.DataFrame(0.0, index=subgroup_proportions.index, columns=['outcome_0', 'outcome_1'])
    
    for idx, row in subgroup_proportions.iterrows():
        subgroup = row['subgroup']
        facet = row['facet']
        subgroup_disparity = row[['outcome_0', 'outcome_1']] - overall_proportions
        subgroup_disparity = pd.Series(subgroup_disparity, index=['outcome_0', 'outcome_1'])
        
        # print(f"Initial subgroup_disparity for subgroup '{subgroup}', facet '{facet}':\n{subgroup_disparity}")
        
        if subgroup_disparity.isnull().values.any():
            print(f"NaN values found in subgroup_disparity for subgroup '{subgroup}', facet '{facet}'. Row data:\n{row}")
            subgroup_disparity = subgroup_disparity.fillna(0)
        
        subgroup_weight = len(df[df[subgroup_column] == subgroup]) / len(df)
        print(f"Subgroup: {subgroup}, Facet: {facet}, Weight: {subgroup_weight}")
        # print(f"Subgroup Disparity:\n{subgroup_disparity}")
        
        # Print detailed information before updating
        # print(f"Before updating, aggregated_disparity:\n{aggregated_disparity.loc[idx]}")
        
        # Compute the weighted disparity
        weighted_disparity = subgroup_disparity * subgroup_weight
        # print(f"Weighted disparity for subgroup '{subgroup}', facet '{facet}':\n{weighted_disparity}")
        
        # Check for NaNs in the weighted disparity
        if weighted_disparity.isnull().values.any():
            print(f"NaNs found in weighted disparity for subgroup '{subgroup}', facet '{facet}'.")
        
        # Update the aggregated disparity
        aggregated_disparity.loc[idx] += weighted_disparity
        
        print(f"After updating, aggregated_disparity:\n{aggregated_disparity.loc[idx]}")
    
    if aggregated_disparity.isnull().values.any():
        print("NaN values found in aggregated_disparity even after handling. This may indicate insufficient data or extreme imbalance.")
        print(f"Aggregated Disparity:\n{aggregated_disparity}")
    
    # Convert back to original indices if needed
    aggregated_disparity.index = subgroup_proportions.set_index(['subgroup', 'facet']).index
    
    return aggregated_disparity


def compute_probability_distributions(df, facet_name, label_column):
    facet_label_counts = df.groupby(facet_name)[label_column].value_counts().unstack(fill_value=0)
    probability_distributions = facet_label_counts.div(facet_label_counts.sum(axis=1), axis=0)
    distributions_dict = {facet: probabilities.values for facet, probabilities in probability_distributions.iterrows()}
    return distributions_dict


def jensen_shannon_divergence(Pa, Pd):
    M = 0.5 * (Pa + Pd)
    kl_pm = kullback_leibler_divergence(Pa, M)
    kl_qm = kullback_leibler_divergence(Pd, M)
    js_divergence = 0.5 * (kl_pm + kl_qm)
    return js_divergence


def lp_norm(Pa, Pd, p=2):
    return np.linalg.norm(Pa - Pd, ord=p)


def generalized_total_variation_distance(df, facet_name, outcome_name):
    outcome_counts = df.groupby([facet_name, outcome_name]).size().unstack(fill_value=0)
    outcome_probabilities = outcome_counts.div(outcome_counts.sum(axis=1), axis=0)

    facets = outcome_probabilities.index.unique()
    n = len(facets)
    
    if n < 2:
        raise ValueError("Not enough groups for comparison (at least two required).")
    
    total_tvd = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            na = outcome_probabilities.loc[facets[i]]
            nb = outcome_probabilities.loc[facets[j]]
            l1_norm = sum(abs(na[k] - nb[k]) for k in na.index)
            tvd = 0.5 * l1_norm
            total_tvd += tvd
            count += 1

    average_tvd = total_tvd / count if count > 0 else 0
    return average_tvd


def kolmogorov_smirnov_metric(Pa, Pd):
    
    if Pa.size == 0 or Pd.size == 0:
        raise ValueError("One or both probability distributions are empty.")
        
    if Pa.size != Pd.size:
        raise ValueError("Distributions must be of the same length.")

    ks_metric = max(abs(Pa - Pd))
    return ks_metric


def normalized_mutual_information(df, facet_name, outcome_name):
    return normalized_mutual_info_score(df[facet_name], df[outcome_name])


def conditional_mutual_information(df, facet_name, outcome_name, conditional_column):
    unique_values = df[conditional_column].unique()
    cond_mi = 0
    for value in unique_values:
        subset = df[df[conditional_column] == value]
        mi = mutual_info_score(subset[facet_name], subset[outcome_name])
        cond_mi += (len(subset) / len(df)) * mi
    return cond_mi / np.log(len(unique_values))


def binary_ratio(df, facet_name, outcome_name):
    outcomes = df.groupby(facet_name)[outcome_name].mean()
    return outcomes[1] / outcomes[0]


def binary_difference(df, facet_name, outcome_name):
    outcomes = df.groupby(facet_name)[outcome_name].mean()
    return outcomes[1] - outcomes[0]


def conditional_binary_difference(df, facet_name, outcome_name, conditional_column):
    unique_values = df[conditional_column].unique()
    cond_diff = 0
    for value in unique_values:
        subset = df[df[conditional_column] == value]
        diff = binary_difference(subset, facet_name, outcome_name)
        cond_diff += (len(subset) / len(df)) * diff
    return cond_diff


def pearson_correlation(df, facet_name, outcome_name):
    return df[facet_name].corr(df[outcome_name])


def logistic_regression_analysis(df, facet_name, outcome_name):
    model = LogisticRegression()
    X = df[facet_name].values.reshape(-1, 1)
    y = df[outcome_name]
    model.fit(X, y)
    return model.coef_, model.intercept_


def get_user_input():
    ''' Interactive method to ask for user settings.'''

    file_path = input("Enter the path to your dataset (CSV or JSON file): ")

    if not file_path.lower().endswith(('.csv', '.json')):
        print("Unsupported file type. Please provide a CSV or JSON file.")
        return None

    use_subgroup_analysis = int(input("Enter 1 to apply subgroup analysis, 0 to skip: "))
    facet_name = input("Enter the column name for the facet (e.g., Gender): ")
    outcome_name = input("Enter the column name for the outcome (e.g., Lymphoma): ")
    subgroup_column = input("Enter the column name for subgroup categorization (optional, press Enter to skip): ")

    try:
        label_value = int(input("Enter the label value or threshold for positive outcomes (e.g., 1): "))
    except ValueError:
        print("Invalid input! Please enter a valid integer for the label value.")
        return None

    return file_path, facet_name, outcome_name, subgroup_column, label_value, use_subgroup_analysis

def get_input(**kwargs):
    ''' 
    get  settings from dictionary
    key: value, required
    file_path: path_to_csv_or_json, yes
    subgroup_analysis: 1 or 0, yes
    facet: col_name, yes
    outcome_name: outcome_col_name, yes
    subgroup_col: demographic_col_name, no
    label_value: label value or threshold, yes
    '''

    file_path = kwargs["file_path"]

    if not file_path.lower().endswith(('.csv', '.json')):
        print("Unsupported file type. Please provide a CSV or JSON file.")
        return None

    use_subgroup_analysis = kwargs["subgroup_analysis"]
    facet_name = kwargs["facet"]
    outcome_name = kwargs["outcome"]
    subgroup_column = ''
    if "subgroup_col" in kwargs:
        subgroup_column = kwargs["subgroup_col"]
    
    try:
        label_value = kwargs["label_value"]
    except ValueError:
        print("Invalid input! Please enter a valid integer for the label value.")
        return None

    return file_path, facet_name, outcome_name, subgroup_column, label_value, use_subgroup_analysis


def load_data(file_path):
    try:
        if file_path.lower().endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.lower().endswith('.json'):
            return pd.read_json(file_path)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None


def calculate_metrics(D, facet_name, label_values_or_threshold, outcome_name, subgroup_column, file_path, cluster_id=None):
    print("Calculating pre-training data bias metrics...")

    metrics = {}

    try:
        # CLASS IMBALANCE (CI)
        ci = calculate_generalized_imbalance(D, facet_name)
        metrics['Class Imbalance (CI)'] = ci
        print("- CI for", facet_name, "is", str(ci))

        if ((np.around(ci, 2) >= 0.9) | (np.around(ci, 2) <= -0.9)):
            print(">> Warning: Significant bias detected based on CI metric!")

        # DIFFERENCE IN PROPORTION LABELS (DPL)
        num_facet = D[facet_name].value_counts()
        num_facet_adv = num_facet.iloc[1] if 1 in num_facet.index else 0
        num_facet_disadv = num_facet.iloc[0] if 0 in num_facet.index else 0
        num_facet_and_pos_label = D[facet_name].where(D[outcome_name] == label_values_or_threshold).value_counts()
        num_facet_and_pos_label_adv = num_facet_and_pos_label.iloc[1] if 1 in num_facet_and_pos_label.index else 0
        num_facet_and_pos_label_disadv = num_facet_and_pos_label.iloc[0] if 0 in num_facet_and_pos_label.index else 0

        dpl = calculate_difference_in_proportions(num_facet_and_pos_label_adv,
                                                  num_facet_adv,
                                                  num_facet_and_pos_label_disadv,
                                                  num_facet_disadv)
        metrics['Difference in Proportion Labels (DPL)'] = dpl
        print("- DPL for", facet_name, "given the outcome", outcome_name, "=", str(label_values_or_threshold), "is", str(dpl))

        if abs(dpl) > 0.1:
            print(">> Warning: Significant bias detected based on DPL metric!")

        # DEMOGRAPHIC DISPARITY (DD)
        dd = generalized_demographic_disparity(D, facet_name, outcome_name, reference_group=None)
        dd_mean = dd.mean().mean()
        metrics['Demographic Disparity (DD)'] = dd_mean
        print("- Average DD for", facet_name, "given the outcome", outcome_name, "is:", str(dd_mean))

        if abs(dd_mean) > 0.1:
            print(">> Warning: Significant bias detected based on DD metric!")

        # CONDITIONAL DEMOGRAPHIC DISPARITY (CDD)
        cdd = None
        if subgroup_column.strip():
            cdd = generalized_conditional_demographic_disparity(D, facet_name, outcome_name, subgroup_column)
            if cdd is not None:
                cdd_mean = cdd.mean().mean()
                metrics['Conditional Demographic Disparity (CDD)'] = cdd_mean
                print("- Average CDD for", facet_name, "given the outcome", outcome_name, "conditioned by", subgroup_column, "is:", str(cdd_mean))
                
                if cdd_mean > 0.1:
                    print(">> Warning: Significant bias detected based on CDD metric!")
            else:
                print(f"CDD calculation for {facet_name} and {outcome_name} conditioned by {subgroup_column} returned None.")
        else:
            print("- Average CDD: Subgroup was not provided.")   

        # Compute the probability distributions for the facets
        probability_distributions = compute_probability_distributions(D, facet_name, outcome_name)
        Pa = probability_distributions.get(1, np.array([]))
        Pd = probability_distributions.get(0, np.array([]))

        js_divergence, l2_norm_value, ks_value, tvd = None, None, None, None
        if Pa.size > 0 and Pd.size > 0 and Pa.size == Pd.size:
            js_divergence = jensen_shannon_divergence(Pa, Pd)
            metrics['Jensen-Shannon Divergence (JS)'] = js_divergence
            print("- Jensen-Shannon Divergence between", facet_name, "and", outcome_name, "is", str(js_divergence))

            if js_divergence > 0.1:
                print(">> Warning: Significant bias detected based on JS metric!")

            l2_norm_value = lp_norm(Pa, Pd, p=2)
            metrics['L2 Norm'] = l2_norm_value
            print("- L2 norm between", facet_name, "and", outcome_name, "is", str(l2_norm_value))

            if l2_norm_value > 0.1:
                print(">> Warning: Significant bias detected based on L2 norm metric!")

            try:
                tvd = generalized_total_variation_distance(D, facet_name, outcome_name)
                metrics['Total Variation Distance (TVD)'] = tvd
                print("- TVD for", facet_name, "given", outcome_name, "is", str(tvd))

                if tvd > 0.1:
                    print(">> Warning: Significant bias detected based on TVD metric!")

            except ValueError as e:
                print(e)

            ks_value = kolmogorov_smirnov_metric(Pa, Pd)
            metrics['KS value'] = ks_value
            print("- KS metric between", facet_name, "and", outcome_name, "is", str(ks_value))
            if abs(ks_value) > 0.1:
                print(">> Warning: Significant bias detected based on KS metric!")
        else:
            print("Cannot compute Jensen-Shannon Divergence, L2 norm, TVD, KS due to data issues.")

        # NORMALIZED MUTUAL INFORMATION (NMI)
        nmi = normalized_mutual_information(D, facet_name, outcome_name)
        metrics['Normalized Mutual Information (NMI)'] = nmi
        print("- NMI between", facet_name, "and", outcome_name, "is", str(nmi))

        # CONDITIONAL MUTUAL INFORMATION (NCMI)
        cond_nmi = None
        if subgroup_column.strip():
            cond_nmi = conditional_mutual_information(D, facet_name, outcome_name, subgroup_column)
            metrics['Conditional Mutual Information (NCMI)'] = cond_nmi
            print("- NCMI for", facet_name, "and", outcome_name, "conditioned on", subgroup_column, "is", str(cond_nmi))
        else:
            print("- NCMI: Subgroup was not provided.")

        # BINARY RATIO (BR)
        ratio, diff, cond_diff, corr, coeffs, intercept = None, None, None, None, None, None
        if D[facet_name].nunique() == 2 and D[outcome_name].nunique() == 2:
            ratio = binary_ratio(D, facet_name, outcome_name)
            metrics['Binary Ratio (BR)'] = ratio
            print("- BR for", facet_name, "and", outcome_name, "is", str(ratio))
        else:
            print("- BR: One or both variables are not binary.")

        # BINARY DIFFERENCE (BD)
        if D[facet_name].nunique() == 2 and D[outcome_name].nunique() == 2:
            diff = binary_difference(D, facet_name, outcome_name)
            metrics['Binary Difference (BD)'] = diff
            print("- BD for", facet_name, "and", outcome_name, "is", str(diff))
        else:
            print("- BD: One or both variables are not binary.")

        # CONDITIONAL BINARY DIFFERENCE (CBD)
        if subgroup_column.strip() and D[facet_name].nunique() == 2 and D[outcome_name].nunique() == 2:
            cond_diff = conditional_binary_difference(D, facet_name, outcome_name, subgroup_column)
            metrics['Conditional Binary Difference (CBD)'] = cond_diff
            print("- CBD for", facet_name, "and", outcome_name, "conditioned on", subgroup_column, "is", str(cond_diff))
        else:
            print("- CBD: Missing conditions for binary conditional difference.")

        # PEARSON CORRELATION (CORR) and CRAMER'S V
        if pd.api.types.is_numeric_dtype(D[facet_name]) and pd.api.types.is_numeric_dtype(D[outcome_name]):
            corr = pearson_correlation(D, facet_name, outcome_name)
            metrics['Pearson Correlation (CORR)'] = corr
            print("- CORR between", facet_name, "and", outcome_name, "is", str(corr))
        else:
            print("- CORR: Variables are not numeric. Calculating Cramér's V instead.")
            cramer_v = cramers_v(D[facet_name], D[outcome_name])
            metrics["Cramér's V"] = cramer_v
            print("- Cramér's V between", facet_name, "and", outcome_name, "is", str(cramer_v))

        # LOGISTIC REGRESSION (LR)
        if D[facet_name].nunique() == 2:
            coeffs, intercept = logistic_regression_analysis(D, facet_name, outcome_name)
            metrics['Logistic Regression Coefficient (LR)'] = coeffs[0][0]
            metrics['Logistic Regression Intercept (Intercept)'] = intercept[0]
            print("- LR coefficients for", facet_name, "predicting", outcome_name, "are", str(coeffs))
            print("  Intercept is", str(intercept))
        else:
            print("- LR: Protected feature is not binary or outcome is not multi-labeled.")

        return metrics

    except ZeroDivisionError as e:
        print("ZeroDivisionError: ", e)
    except Exception as e:
        print("Error: ", e)
        return None

def _check_values(config:dict):
    '''Checks for path, facet and output. Not if columns exists.'''
    if  not config["file_path"]:
        raise ValueError( "file path (-f {path}) or plain interactive (-i) required")

    if not os.path.exists(config["file_path"]):
        raise FileExistsError(f"Cannot find {config['file_path']}")
    
    if not config["facet"]:
        raise ValueError( 
            "Column name for the facet (-t {col_name}) or plain interactive (-i) required"
        )
    
    if not config["outcome"]:
        raise ValueError( 
            "Column name of the outcome (-o {col_name}) or plain interactive (-i) required"
        )

def bias_config(
        interactive:bool=False,
        file_path:str='',
        subgroup_analysis:int=0,
        facet:str='',
        outcome:str='',
        subgroup_col:str='',
        label_value:float=1.0,
        config:dict={}
    )->dict:
    '''Main function for bias analysis'''
    
    if interactive:
        user_input = get_user_input()

    elif config and isinstance(config,dict):
        user_input = get_input(**config)

    else:

        config = {
            "file_path": file_path,
            "facet": facet,
            "outcome": outcome,
            "subgroup_col": subgroup_col,
            "label_value": label_value,
            "subgroup_analysis": subgroup_analysis,
        }

        _check_values(config=config)

        user_input = [value for value in config.values()]

    file_path, facet_name, outcome_name, subgroup_column, label_values_or_threshold, use_subgroup_analysis = user_input
    D = load_data(file_path)
    if D is None:
        return
    
    D, mappings = map_categorical_to_integers(D)
    print("Categorical feature mappings:")
    for feature, mapping in mappings.items():
        print(f"{feature}: {mapping}")
    
    D, age_mapping = group_ages(D)
    if age_mapping:
        print("Age group mapping:")
        print(age_mapping)

    # Changed output location so results would not mix with inputs.
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join("./Materials/BiasAssessment",file_name)
    os.makedirs(output_dir, exist_ok=True)

    # out_csv_path = os.path.join(output_dir, file_name+'_coded.csv')
    # D.to_csv(out_csv_path, index=False)

    all_metrics = {}

    # Same here
    png_out_path = os.path.join(output_dir, file_name)

    if use_subgroup_analysis == 1:
        max_clusters = 30
        cluster_counts, db_scores, optimal_clusters = find_optimal_clusters_db(D, max_clusters)
        print(f"The optimal number of clusters is {optimal_clusters}")
        plot_db_scores(cluster_counts, db_scores, optimal_clusters)

        _, cluster_map, _ = perform_clustering(D, optimal_clusters)

        for ct, (cluster_id, indices) in enumerate(cluster_map.items(), start=1):
            try:
                print(f"\nStarting analysis for cluster {ct} of {optimal_clusters} with {len(indices)} samples.")
                Dk = D.iloc[indices]

                if len(np.unique(Dk[outcome_name])) == 1 or len(np.unique(Dk[facet_name])) == 1:
                    print(f"Skipping cluster {ct}: Not enough diversity in '{outcome_name}' or in '{facet_name}'.")
                else:
                    try:
                        cluster_metrics = calculate_metrics(Dk, facet_name, label_values_or_threshold, outcome_name, subgroup_column, file_path, cluster_id=cluster_id)
                        if cluster_metrics:
                            all_metrics[f'Cluster {ct}'] = cluster_metrics
                            # Plot metrics for this cluster
                            plot_bias_metrics(cluster_metrics, png_out_path, cluster_id=ct)
                    except Exception as e:
                        print(f"Failed to calculate metrics for cluster {ct}: {e}")
            except Exception as e:
                print(f"Error processing cluster {ct}: {e}")
        print("Done")
    else:
        try:
            overall_metrics = calculate_metrics(D, facet_name, label_values_or_threshold, outcome_name, subgroup_column, file_path)
            if overall_metrics:
                all_metrics['Overall'] = overall_metrics
                # Plot overall metrics
                plot_bias_metrics(overall_metrics, png_out_path)
        except Exception as e:
            print(f"Failed to calculate metrics: {e}")

    # Save all metrics to JSON
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    json_file_path = file_name + '_all_metrics.xlsx'
    am = pd.DataFrame(all_metrics)
    json_file_path = os.path.join(output_dir, json_file_path)
    am.to_excel(json_file_path)
    # with open(json_file_path, 'w') as json_file:
    #     json.dump(all_metrics, json_file, indent=4)
    print(f"All metrics saved to {json_file_path}")
    
    # added return so not to reload the dictionary from json, into passport
    return all_metrics

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    config = {
        "file_path": '',
        "subgroup_analysis":'',
        "facet":'',
        "outcome":'',
        "subgroup_col": '',
        "label":''
    }

    parser.add_argument('--file_path', '-f', type=str, help= "Path to csv or json file", default="")
    
    parser.add_argument(
        '--subgroup_analysis', '-a', type=int, help="1: apply subgroup analysis, 0: skip", default=0
    )
    
    parser.add_argument(
        '--facet', '-t', type=str, help="Column name for the facet (e.g., Gender)", default="")
    
    parser.add_argument(
        '--outcome', '-o', type=str, help="column name for the outcome (e.g., Lymphoma)"
    )
    
    parser.add_argument(
        "--subgroup_col", '-c', type=str,
        help= "Column name for subgroup categorization (optional)", default= ""
    )
    
    parser.add_argument(
        '--label_value', '-l', type=float, help="Label value or threshold", default=1
    )

    parser.add_argument(
        '--interactive','-i', action='store_true', help='Activate interactive session'
    )

    args = parser.parse_args()

    if args.interactive:
        bias_config(interactive=True)

    else:

        config = {
            "file_path": args.file_path,
            "subgroup_analysis": args.subgroup_analysis,
            "facet": args.facet,
            "outcome": args.outcome,
            "subgroup_col": args.subgroup_col,
            "label_value": args.label_value,
        }

        _check_values(config=config)

        bias_config(bias_config=config)