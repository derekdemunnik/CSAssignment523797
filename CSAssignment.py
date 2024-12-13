import json
import re
import math
import copy
import numpy as np
import sympy
import random
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from strsimpy.qgram import QGram
import textdistance
from collections import defaultdict


# Part 1:Datacleaning ---------------------------------------------------------------

def clean_data(value):
    # Convert to lower case first
    value = value.lower()

    # Normalize variations of "inch", "hertz", "lbs", "mw", "wifi", and "year"
    inch_variations = [r'["”]', r'inch', r'inches', r'-inch', r' inch']
    for var in inch_variations:
        value = re.sub(var, 'inch', value)

    hertz_variations = [r'hz', r'hertz', r'-hz', r' hz']
    for var in hertz_variations:
        value = re.sub(var, 'hz', value)

    pounds_variations = [r'pounds', r' pounds', r'lb', r' lbs', r'lbs.']
    for var in pounds_variations:
        value = re.sub(var, 'lbs', value)
        
    year_variations = [r' year limited', r' years limited', r' years', r' year', r' Year', r'Year', r'year']
    for var in year_variations:
        value = re.sub(var, 'year', value)
    
    # Function to remove non-alphanumeric tokens and spaces before units
    def clean_unit(match):
        unit = match.group(0)
        return re.sub(r'^[\s\W]+', '', unit)

    # Apply the cleaning function to each instance inch, hertz, pounds, year
    value = re.sub(r'[\s\W]*(inch|hz|lbs|year)\b', clean_unit, value)

    return value

def clean_tv_dataset(file_path, output_file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Create a deep copy of the original data
    data_cleaned = copy.deepcopy(data)

    for model_id, list_tvs in data_cleaned.items():
        for tv in list_tvs:
            if 'featuresMap' in tv:
                for feature, v in tv['featuresMap'].items():
                    cleaned_val = clean_data(v)
                    tv['featuresMap'][feature] = cleaned_val
            if 'title' in tv:
                tv['title'] = clean_data(tv['title'])

    # Save the cleaned data to a new JSON file
    with open(output_file_path, 'w') as file:
        json.dump(data_cleaned, file, indent=4)

input_file_path = "TVs-all-merged.json"
output_file_path = "TVs-all-merged-new.json"
clean_tv_dataset(input_file_path, output_file_path)


# Part 2: Extracting model words from titles and Key-value pairs to create binary vectors ---------------------------------------------------------------

def get_mwtitles(title):
    model_words_title = []
    pattern = r'([a-zA-Z0-9]*(([0-9]+[^0-9, ]+)|([^0-9, ]+[0-9]+))[a-zA-Z0-9]*)'
    regex = re.compile(pattern)
    matches = regex.findall(title)
    for match in matches:
        model_words_title.append(match[0])
    return model_words_title

def get_uniqmw(titles_list):
    all_model_words_title = []
    for title in titles_list:
        model_words = get_mwtitles(title)
        all_model_words_title.extend(model_words)
    return list(set(all_model_words_title))

def get_mwvalues(value):
    model_words_keyvalue = []
    pattern = r'(^\d+(\.\d+)?[a-zA-Z]*$|^\d+(\.\d+)?$)'
    regex = re.compile(pattern)
    matches = regex.findall(value)
    for match in matches:
        numeric_part = re.match(r'^\d+(\.\d+)?', match[0])
        if numeric_part:
            model_words_keyvalue.append(numeric_part.group())
    return model_words_keyvalue

def get_uniqmwfeat(features_list):
    all_model_words_feature = []
    for feature in features_list:
        model_words = get_mwvalues(feature)
        all_model_words_feature.extend(model_words)
    return list(set(all_model_words_feature))

def get_binmatrix(products_subset):
    titles_list = [product['title'] for product in products_subset]
    unique_model_words_title = set(get_uniqmw(titles_list))
    features_values = [value for product in products_subset for value in product.get('featuresMap', {}).values()]
    unique_model_words_features = set(get_uniqmwfeat(features_values))

    all_mw = set(unique_model_words_title.union(unique_model_words_features))
    binary_matrix = np.zeros((len(all_mw), len(products_subset)), dtype=int)

    for product_index, product in enumerate(products_subset):
        title = product['title']
        features_values = product.get('featuresMap', {}).values()

        for mw_index, mw in enumerate(all_mw):
            if (mw in (unique_model_words_title or unique_model_words_features) and mw in features_values) or \
               (mw in (unique_model_words_title) and mw in title):
                binary_matrix[mw_index][product_index] = 1

    return binary_matrix
  

# Part3: Create signature matrix using minhashing ---------------------------------------------------------------

def hash_function(a, b, x, p):
    """
    Hash function used for minhashing.
    """
    return (a + b * x) % p

def minhashing(binary_matrix): 
    """
    Generate the signature matrix using minhashing.
    """
    num_rows, num_cols = binary_matrix.shape

    # Set k to a fixed value or percentage of rows
    k = 700  # Number of hash functions/signature rows

    p = sympy.nextprime(num_rows)  # Prime number greater than the number of rows

    # Generate random coefficients for the hash functions
    a_values = np.array([random.randint(1, p) for _ in range(k)])
    b_values = np.array([random.randint(0, p) for _ in range(k)])

    # Initialize the signature matrix with maximum integer values
    signature_matrix = np.full((k, num_cols), np.iinfo(np.int32).max, dtype=np.int32)

    # Populate the signature matrix
    for r in range(num_rows):
        # Compute all hash values for the current row
        hf = [hash_function(a_values[i], b_values[i], r, p) for i in range(k)]
        for c in range(num_cols): 
            if binary_matrix[r][c] == 1:
                for j in range(k):
                    signature_matrix[j][c] = min(signature_matrix[j][c], hf[j])

    return signature_matrix

#True Duplicates and Signature Matrix Generation
def get_duplicateset(list_products):
    """
    Identify true duplicate pairs based on modelID.
    """
    # Dictionary to hold the products grouped by their modelID
    true_duplicates = defaultdict(list)

    # Group products by modelID
    for idx, product in enumerate(list_products):
        true_duplicates[product['modelID']].append(idx)

    # Generate pairs of duplicates
    true_pairs = set()
    for indices in true_duplicates.values():
        if len(indices) > 1:
            for pair in combinations(indices, 2):
                true_pairs.add(tuple(sorted(pair)))

    return true_pairs

def get_duplicateset_subset(list_products, original_indices):
    """
    Identify true duplicate pairs in a subset using original indices.
    """
    # Dictionary to hold the products grouped by their modelID
    true_duplicates = defaultdict(list)

    # Group subset products by modelID
    for original_idx, product in zip(original_indices, list_products):
        true_duplicates[product['modelID']].append(original_idx)

    # Generate pairs of duplicates
    true_pairs = set()
    for indices in true_duplicates.values():
        if len(indices) > 1:
            for pair in combinations(indices, 2):
                true_pairs.add(tuple(sorted(pair)))

    return true_pairs

def get_signature_matrix(products):
    """
    Generate a signature matrix and true duplicate pairs for the product list.
    """
    pairs = list(combinations(range(len(products)), 2))  # All possible pairs
    true_pairs = get_duplicateset(products)  # Identify true duplicate pairs
    binary_matrix = get_binmatrix(products)  # Obtain the binary matrix
    signature_matrix = minhashing(binary_matrix)  # Generate minhashing signature matrix
    rows, cols = signature_matrix.shape
    n = rows  # Number of hash functions

    return pairs, true_pairs, signature_matrix, n
    

# Part4: LSH method  ---------------------------------------------------------------

def LSH(sig_matrix, b, r):
    
    n, num_products = sig_matrix.shape
    assert n == b * r, "Number of rows in signature matrix must equal b * r"

    # Dictionary to store buckets for each band
    buckets = {}

    for i in range(b):
        for j in range(num_products):
            # Create hashable tuple key for each column in the band
            key = tuple(sig_matrix[i * r:(i + 1) * r, j])
            buckets.setdefault(key, []).append(j)
    
    # Initialize the set of candidate pairs
    candidate_pairs = set()

    # Generate pairs from each bucket
    for bucket in buckets.values():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    pair = (min(bucket[i], bucket[j]), max(bucket[i], bucket[j]))
                    candidate_pairs.add(pair)

    return candidate_pairs

# Part 5:MSM method---------------------------------------------------------------


def cosinesimilarity(a, b):
    set_a = set(a.split())
    set_b = set(b.split())
    
    # Calculate the intersection of the sets
    intersection = set_a.intersection(set_b)
    # Apply the formula |a ∩ b| / (√|a| * √|b|)
    nameCosineSim = len(intersection) / (len(set_a) ** 0.5 * len(set_b) ** 0.5)
    return nameCosineSim 

def breakmw(model_word):
    non_numeric = re.sub(r'[0-9]', '', model_word)
    numeric = re.sub(r'[^0-9]', '', model_word)
    return non_numeric, numeric

def avgLvSim(set_X, set_Y):
    total_similarity = 0
    total_length = 0
    
    for x in set_X:
        for y in set_Y:
            # Calculate the normalized Levenshtein similarity as 1 - normalized distance
            similarity = 1 - textdistance.levenshtein.normalized_distance(x, y)
            # Weight the similarity by the sum of lengths of x and y
            weight = len(x) + len(y)
            total_similarity += similarity * weight
            total_length += weight

    # The overall average is the total weighted similarity divided by the total length
    return total_similarity / total_length if total_length else 0

def avgLvSimMW(X, Y): 
    similarity_sum = 0
    denominator = 0
    for x in X:
        non_numeric_x, numeric_x = breakmw(x)
        for y in Y:
            non_numeric_y, numeric_y = breakmw(y)
            if textdistance.levenshtein.normalized_similarity(non_numeric_x, non_numeric_y) > 0.7 and numeric_x == numeric_y:
                similarity = (1 - textdistance.levenshtein.normalized_distance(x, y)) * (len(x) + len(y))
                similarity_sum += similarity
                denominator += len(x) + len(y)
            
    avg_similarity = similarity_sum / denominator if denominator != 0 else 0

    return avg_similarity

def TMWMSim(p1, p2, alpha, beta, delta, epsilon): 
    # p1, p2 is title (str)
    nameCosineSim = cosinesimilarity(p1, p2)
    if nameCosineSim > alpha:
        return 1
    
    modelWordsA = set(get_mwtitles(p1))
    modelWordsB = set(get_mwtitles(p2))

    for wordA in modelWordsA: 
        for wordB in modelWordsB: 
            non_numericA, numericA = breakmw(wordA)
            non_numericB, numericB = breakmw(wordB)
            if numericA != numericB and textdistance.levenshtein.normalized_similarity(non_numericA, non_numericB) > 0.5:
                return -1
     
    finalNameSim = beta * nameCosineSim + (1 - beta) * avgLvSim(modelWordsA, modelWordsB)

    similarityCheck = False
    for wordA in modelWordsA: 
        for wordB in modelWordsB: 
            non_numericA, numericA = breakmw(wordA)
            non_numericB, numericB = breakmw(wordB)
            if numericA == numericB and textdistance.levenshtein.normalized_similarity(non_numericA, non_numericB) > 0.5:
                similarityCheck = True
    
    if similarityCheck == True:
        modelWordSimVal = avgLvSimMW(modelWordsA, modelWordsB)
        finalNameSim = delta * modelWordSimVal + (1-delta) * finalNameSim
    
    if finalNameSim > epsilon: 
        return finalNameSim
    else:
        return -1
    
def calcSim(string1, string2):
    qgram = QGram(3)
    n1 = len((string1))
    n2 = len((string2))
    valueSim = (n1 + n2 - qgram.distance(string1, string2))/ (n1 + n2)
    return valueSim

def get_brand_or_brand_name(features_map):
    # Try to get 'Brand', if not found, try 'Brand Name'
    return features_map.get('Brand') or features_map.get('Brand Name')

def mw(C, D):
    intersection = len(set(C) & set(D))
    union = len(set(C) | set(D))

    return (intersection / union) if union else 0

def exMW(p):
    return set(get_uniqmwfeat([v for v in p.values()]))

def aggclustering(dist_matrix, threshold):
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        linkage='single', 
        distance_threshold=threshold, 
        metric='precomputed'
    )
    clusters = clustering.fit_predict(dist_matrix)
    return clusters

def MSM(product_list, pairs, gamma, alpha, beta, mu, delta, epsilon_TMWM, epsilon_clustering):
    n = len(product_list)
    large_number = 1e10
    dist = np.full((n, n), large_number)
    for pi_x, pj_y in pairs: 
        pi = product_list[pi_x]
        pj = product_list[pj_y]
        
        pi_brand = get_brand_or_brand_name(pi['featuresMap'])
        pj_brand = get_brand_or_brand_name(pj['featuresMap'])

        if pi['shop'] == pj['shop'] or (pi_brand != pj_brand and pi_brand is not None and pj_brand is not None):
            continue

        else: 
            sim = 0
            avgSim = 0
            m = 0 
            w = 0
            
            keys_i = set(pi['featuresMap'].keys())
            keys_j = set(pj['featuresMap'].keys())

            nmk_i_keys = keys_i - keys_j
            nmk_j_keys = keys_j - keys_i

            nmk_i = {k: pi['featuresMap'][k] for k in nmk_i_keys}
            nmk_j = {k: pj['featuresMap'][k] for k in nmk_j_keys}


            for key_i, val_i in pi['featuresMap'].items():
                for key_j, val_j in pj['featuresMap'].items():
                    keySim = calcSim(key_i, key_j)
                    if keySim > gamma: 
                        valueSim = calcSim(val_i, val_j)
                        weight = keySim
                        sim = sim + weight * valueSim
                        m = m + 1
                        w = w + weight
                        nmk_i.pop(key_i, None)
                        nmk_j.pop(key_j, None)
            
            if w > 0:
                avgSim = sim / w
            
            mwPerc = mw(exMW(nmk_i), exMW(nmk_j))
            titleSim = TMWMSim(pi['title'], pj['title'], alpha, beta, delta, epsilon_TMWM)

            if titleSim == -1:
                minFeatures = min(len(pi['featuresMap']), len(pj['featuresMap']))
                theta1 = m/minFeatures
                theta2 = 1-theta1
                hSim = theta1 * avgSim + theta2 * mwPerc
            else: 
                theta1 = (1-mu)*(m / min(len(pi['featuresMap']), len(pj['featuresMap'])))
                theta2 = 1 - mu - theta1
                hSim = theta1 * avgSim + theta2 * mwPerc + mu * titleSim
                if pj_y == int(pi_x + 1):
                    print(pi_x, pj_y)
                    print(f"hSim: {hSim}, theta1: {theta1}, avgSim: {avgSim}, theta2: {theta2}, mwPerc: {mwPerc}, titleSim: {titleSim}")
            
            dist[pi_x][pj_y] = 1-hSim
    
    return aggclustering(dist, epsilon_clustering)




# Part 6: Distance Matrix Generation ---------------------------------------------------------------

#Function to generate the distance matrix
def generate_distance_matrix_opt(full_product_list, all_pairs, gamma, alpha, beta, mu, delta, epsilon_TMWM):
    n = len(full_product_list)
    large_number = 1e10
    dist = np.full((n, n), large_number)
    count_pairs = 0

    # Preprocessing step
    precomputed_data = {
        i: {
            'brand': get_brand_or_brand_name(pi['featuresMap']),
            'shop': pi['shop'],
            'title': pi['title'],
            'featuresMap': pi['featuresMap'],
            'featuresKeys': set(pi['featuresMap'].keys())
        } for i, pi in enumerate(full_product_list)
    }
    
    for pi_x, pj_y in all_pairs: 
        count_pairs = count_pairs + 1
        if count_pairs % 5000 == 0: 
            print(count_pairs)
        data_i = precomputed_data[pi_x]
        data_j = precomputed_data[pj_y]

        if data_i['shop'] == data_j['shop'] or (data_i['brand'] != data_j['brand'] and data_i['brand'] is not None and data_j['brand'] is not None):
            continue
            
        else: 
            sim = 0
            avgSim = 0
            m = 0 
            w = 0
            
            keys_i = data_i['featuresKeys']
            keys_j = data_j['featuresKeys']

            nmk_i_keys = keys_i - keys_j
            nmk_j_keys = keys_j - keys_i

            nmk_i = {k: data_i['featuresMap'][k] for k in nmk_i_keys}
            nmk_j = {k: data_j['featuresMap'][k] for k in nmk_j_keys}

            keys_to_remove_i = set()
            keys_to_remove_j = set()

            for key_i, val_i in data_i['featuresMap'].items():
                for key_j, val_j in data_j['featuresMap'].items():
                    keySim = calcSim(key_i, key_j)
                    if keySim > gamma: 
                        valueSim = calcSim(val_i, val_j)
                        weight = keySim
                        sim = sim + weight * valueSim
                        m = m + 1
                        w = w + weight
                        
                        keys_to_remove_i.add(key_i)
                        keys_to_remove_j.add(key_j)
            
            nmk_i = {k: v for k, v in nmk_i.items() if k not in keys_to_remove_i}
            nmk_j = {k: v for k, v in nmk_j.items() if k not in keys_to_remove_j}
            
            if w > 0:
                avgSim = sim / w
            
            mwPerc = mw(exMW(nmk_i), exMW(nmk_j))
            titleSim = TMWMSim(data_i['title'], data_j['title'], alpha, beta, delta, epsilon_TMWM)

            if titleSim == -1:
                minFeatures = min(len(data_i['featuresKeys']), len(data_j['featuresKeys']))
                theta1 = m/minFeatures
                theta2 = 1-theta1
                hSim = theta1 * avgSim + theta2 * mwPerc
            else: 
                theta1 = (1-mu)*(m / min(len(data_i['featuresKeys']), len(data_j['featuresKeys'])))
                theta2 = 1 - mu - theta1
                hSim = theta1 * avgSim + theta2 * mwPerc + mu * titleSim
            
            dist[pi_x][pj_y] = 1-hSim
    

    return dist


def generate_distance_matrix_candidates(full_distance_matrix, candidate_pairs): 
    n = full_distance_matrix.shape[0]
    large_number = 1e10
    dist = np.full((n, n), large_number)

     # Convert the list of pairs into two arrays: rows and cols
    rows, cols = zip(*candidate_pairs)
    # Update the matrix in a vectorized manner
    dist[rows, cols] = full_distance_matrix[rows, cols]
    return dist

#Function to calculate the F1 sores after the MSM algorithm and the distance matrix
def calculate_F1_score(true_pairs, epsilon, distance_matrix):
    clusters = aggclustering(distance_matrix, epsilon)
    cluster_pairs = set()
    for cluster_label in set(clusters):
        product_indices = [i for i, x in enumerate(clusters) if x == cluster_label]
        if len(product_indices) > 1:
            for pair in combinations(product_indices, 2):
                cluster_pairs.add(tuple(sorted(pair)))

    normalized_true_pairs = {tuple(sorted(pair)) for pair in true_pairs}

    # Calculate TP, FP, and FN using the normalized pairs
    TP = len(cluster_pairs.intersection(normalized_true_pairs))
    print(f"TP: {TP}")
    FP = len(cluster_pairs - normalized_true_pairs)
    print(f"FP: {FP}")
    FN = len(normalized_true_pairs - cluster_pairs)
    print(f"FN: {FN}")

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    F_1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    print(f"F1: {F_1}")
    return F_1




# Part 8: Evaluate LSH Performance ---------------------------------------------------------------


# Load and process the cleaned dataset
with open(output_file_path, 'r') as file: 
    data_cleaned = json.load(file)

# Flatten all products into a single list
all_products = [item for model_id, items in data_cleaned.items() for item in items]

# Generate all possible product pairs and true duplicate pairs
all_pairs = list(combinations(range(len(all_products)), 2))
all_true_pairs = get_duplicateset(all_products)

# Generate signature matrix and related data
pairs, true_pairs, signature_matrix, n = get_signature_matrix(all_products)

# Initialize variables for LSH performance evaluation
num_rows_sig, num_cols_sig = signature_matrix.shape
total_comparisons = (num_cols_sig * (num_cols_sig - 1)) / 2
D_n = len(true_pairs)

# Lists to store performance metrics
PQ_values = []
PC_values = []
F1_values = []
fraction_of_comparisons_values = []

# Iterate through possible numbers of bands
for b in range(1, n):
    if n % b != 0:  # Ensure n is divisible by b
        continue
    
    r = n // b  # Calculate rows per band
    candidate_pairs = LSH(signature_matrix, b, r)  # Perform LSH

    # Calculate metrics
    N_c = len(candidate_pairs)
    D_f = len(candidate_pairs.intersection(true_pairs))

    PQ = D_f / N_c if N_c != 0 else 0
    PC = D_f / D_n
    F_1 = (2 * (PQ * PC)) / (PQ + PC) if (PQ + PC) != 0 else 0
    fraction_of_comparisons = N_c / total_comparisons

    # Store metrics
    PQ_values.append(PQ)
    PC_values.append(PC)
    F1_values.append(F_1)
    fraction_of_comparisons_values.append(fraction_of_comparisons)

# Convert lists to numpy arrays for plotting
PQ_values = np.array(PQ_values)
PC_values = np.array(PC_values)
F1_values = np.array(F1_values)
fraction_of_comparisons_values = np.array(fraction_of_comparisons_values)

# Plot performance metrics
# PQ Plot
plt.figure(figsize=(7, 6))
plt.plot(fraction_of_comparisons_values, PQ_values, label='Pair Quality (PQ)', marker='o', color ='black')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('(PQ)')
plt.title('Pair Quality')
plt.grid()
plt.legend()
plt.show()

# PC Plot
plt.figure(figsize=(7, 6))
plt.plot(fraction_of_comparisons_values, PC_values, label='Pair Completeness (PC)', marker='o', color='black')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('(PC)')
plt.title('Pair Completeness')
plt.grid()
plt.legend()
plt.show()

# F1 Score Plot
plt.figure(figsize=(7, 6))
plt.plot(fraction_of_comparisons_values, F1_values, label='F1* Score', marker='o', color='black')
plt.xlabel('Fraction of Comparisons')
plt.ylabel('F1* Score')
plt.title('F1* Score')
plt.grid()
plt.legend()
plt.show()

# Part 9: MSM Evaluation ---------------------------------------------------------------

# Parameters for MSM
alpha = 0.6
beta = 0.35
gamma = 0.75
delta = 0.7
mu = 0.65
epsilon_TMWM = 0
#chosen based on past papers
epsilon_range = [0.35, 0.45, 0.55]

# Bootstrap and train/test split parameters
bootstraps = 5
train_ratio = 0.63

# Load and preprocess data
with open(output_file_path, 'r') as file:
    data_cleaned = json.load(file)

all_products = [item for model_id, items in data_cleaned.items() for item in items]
all_pairs = list(combinations(range(len(all_products)), 2))
all_true_pairs = get_duplicateset(all_products)

# Generate the full distance matrix once
full_distance_matrix = generate_distance_matrix_opt(
    all_products, all_pairs, gamma, alpha, beta, mu, delta, epsilon_TMWM
)

np.save("distance_matrix.npy", full_distance_matrix)
full_distance_matrix = np.load("distance_matrix.npy")

# Initialize scores for training and testing
F1_training_scores = {i: {} for i in range(bootstraps)}
F1_testing_scores = {i: {} for i in range(bootstraps)}

for i in range(bootstraps):
    num_products = len(all_products)
    subset_size = int(num_products * train_ratio)

    # Create train/test subsets
    train_subset = random.sample(all_products, subset_size)
    test_subset = [product for product in all_products if product not in train_subset]

    # Map indices for train and test subsets
    train_original_indices = [all_products.index(p) for p in train_subset]
    test_original_indices = [all_products.index(p) for p in test_subset]

    # Generate candidate pairs for training
    train_pairs, train_true_pairs, train_signature_matrix, n = get_signature_matrix(train_subset)
    train_candidate_pairs = LSH(train_signature_matrix, b=n//2, r=2)

    # Generate candidate pairs for testing
    test_pairs, test_true_pairs, test_signature_matrix, _ = get_signature_matrix(test_subset)
    test_candidate_pairs = LSH(test_signature_matrix, b=n//2, r=2)

    # Map to original indices
    train_candidate_pairs_original = {(train_original_indices[i], train_original_indices[j]) for i, j in train_candidate_pairs}
    test_candidate_pairs_original = {(test_original_indices[i], test_original_indices[j]) for i, j in test_candidate_pairs}

    # Create sparse distance matrices for train and test
    train_distance_matrix = generate_distance_matrix_candidates(full_distance_matrix, train_candidate_pairs_original)
    test_distance_matrix = generate_distance_matrix_candidates(full_distance_matrix, test_candidate_pairs_original)

    # Evaluate F1 scores over the epsilon range for training
    for epsilon_clustering in epsilon_range:
        F1_training_scores[i][epsilon_clustering] = calculate_F1_score(train_true_pairs, epsilon_clustering, train_distance_matrix)

    # Find optimal epsilon for training
    optimal_epsilon = max(F1_training_scores[i], key=F1_training_scores[i].get)

    # Evaluate F1 score for testing with the optimal epsilon
    F1_testing_scores[i] = {
        'F1': calculate_F1_score(test_true_pairs, optimal_epsilon, test_distance_matrix),
        'fraction_of_comparisons': len(test_candidate_pairs_original) / (len(test_subset) * (len(test_subset) - 1) / 2)
    }

# Aggregate results for plotting
def aggregate_results(scores):
    averaged_results = {}
    for i, eps_data in scores.items():
        for epsilon, value in eps_data.items():
            if epsilon not in averaged_results:
                averaged_results[epsilon] = {'total_F1': 0, 'count': 0}
            averaged_results[epsilon]['total_F1'] += value
            averaged_results[epsilon]['count'] += 1

    for epsilon, data in averaged_results.items():
        data['average_F1'] = data['total_F1'] / data['count']

    return averaged_results

training_results = aggregate_results(F1_training_scores)
testing_results = F1_testing_scores

