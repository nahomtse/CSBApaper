import re
import json
import numpy as np
import random
import sys
from math import comb



def standardize_data(data):
    # Declare common value representations to be replaced by the last value of the list.
    #expr = '[a-zA-Z0-9.]*[0-9]+[a-zA-Z0-9.]*'

    # Standardize Data.
    std_data = []
    for modelID in data:
        for incident in data[modelID]:
            incident['title'] = incident['title'].replace('"', 'inch')
            incident['title'] = incident['title'].replace('inches', 'inch')
            incident['title'] = incident['title'].replace('-inch', 'inch')
            incident['title'] = incident['title'].replace('Hertz', 'hz')
            incident['title'] = incident['title'].replace('-hz', 'hz')
            incident['title'] = incident['title'].lower()
            incident['title'] = re.sub("[^a-zA-Z0-9\s\.]","", incident['title']) #remove non-alphabetical characters

    # Standardize features
            for feature in incident["featuresMap"]:
                incident["featuresMap"][feature] = incident["featuresMap"][feature].replace('"', 'inch')
                incident['title'] = incident['title'].replace('inches', 'inch')
                incident['title'] = incident['title'].replace('-inch', 'inch')
                incident['title'] = incident['title'].replace('Hertz', 'hz')
                incident['title'] = incident['title'].replace('-hz', 'hz')
                incident["featuresMap"][feature] = incident["featuresMap"][feature].lower()
                incident["featuresMap"][feature] = re.sub("[^a-zA-Z0-9\s\.]","", incident["featuresMap"][feature])
            std_data.append(incident)

    return std_data


def duplicates_matrix(std_data):
    # Set all values zero initially and add 1 if row and column correspond
    duplicates = np.zeros((len(std_data), len(std_data)))
    for row in range(len(std_data)):
        model_row = std_data[row]["modelID"]
        for col in range(row + 1, len(std_data)):
            model_col = std_data[col]["modelID"]
            if model_row == model_col:
                duplicates[row][col] = 1
                duplicates[col][row] = 1

    return duplicates.astype(int)


def calc_bin_vector(std_data):
    # List of common count features.
    freq_words = ["Aspect Ratio", "UPC", "HDMI", "Component", "Video", "Contrast", "Composite", "Speakers", "HDMI", "USB"]

    model_words = dict()
    binary_vec = []

    # Loop through all incidents to find model words.
    for i in range(len(std_data)):
        incident = std_data[i]
        model_title = re.findall(
            "(?:^|(?<=[ \[\(]))([a-zA-Z0-9]*(?:(?:[0-9]+[^0-9\., ()]+)|(?:[^0-9\., ()]+[0-9]+)|(?:([0-9]+\.[0-9]+)["
            "^0-9\., ()]+))[a-zA-Z0-9]*)(?:$|(?=[ \)\]]))",
            incident["title"])
        incident_mw = []
        for match in model_title:
            if match[0] != '':
                incident_mw.append(match[0])
            else:
                incident_mw.append(match[1])

        # Find model words in the key-value pairs.
        features = incident["featuresMap"]
        for key in features:
            value = features[key]

            # Find decimals.
            # ([0-9]+\.[0-9]+) matches any (numeric) - . - (numeric) - (non-numeric) combination (i.e., decimals).
            # [a-zA-Z0-9]* matches any alphanumeric character (zero or more times).
            mw_decimal = re.findall("([0-9]+\.[0-9]+)[a-zA-Z]*", value)
            for decimal in mw_decimal:
                incident_mw.append(decimal)

            # Group some common features.
            key_mw = key
            for feature in freq_words:
                if feature.lower() in key.lower():
                    key_mw = feature
                    break

            # Find the count value and construct a model word by appending the count to the key.
            if key in freq_words:
                counts = re.findall("^[0-9]+", value)
                for count in counts:
                    if count is not None:
                        incident_mw.append(count + key_mw)

        # Loop through all identified model words and update the binary vector product representation.
        for mw in incident_mw:
            if mw in model_words:
                # Set index for model word to one.
                row = model_words[mw]
                binary_vec[row][i] = 1
            else:
                # Add model word to the binary vector, and set index to one.
                binary_vec.append([0] * len(std_data))
                binary_vec[len(binary_vec) - 1][i] = 1

                # Add model word to the dictionary.
                model_words[mw] = len(binary_vec) - 1
    return binary_vec


from sympy import nextprime


def minhash(binary_vec, n):

    random.seed()

    r = len(binary_vec)
    c = len(binary_vec[0])
    binary_vec = np.array(binary_vec)

    # Find k.
    k = nextprime(r - 1)

    # Generate n random hash functions.
    hash_params = np.empty((n, 2))
    for i in range(n):
        # Generate a, b, and k.
        a = random.randint(1, k - 1)
        b = random.randint(1, k - 1)
        hash_params[i, 0] = a
        hash_params[i, 1] = b

    # Initialize signature matrix to infinity for each element.
    signature = np.full((n, c), np.inf)

    # Loop through the binary vector representation matrix once, to compute the signature matrix.
    for row in range(1, r + 1):
        # Compute each of the n random hashes once for each row.
        e = np.ones(n)
        row_vec = np.full(n, row)
        x = np.stack((e, row_vec), axis=1)
        row_hash = np.sum(hash_params * x, axis=1) % k

        for i in range(n):
            # Update column j if and only if it contains a one and its current value is larger than the hash value for
            # the signature matrix row i.
            updates = np.where(binary_vec[row - 1] == 0, np.inf, row_hash[i])
            signature[i] = np.where(updates < signature[i], row_hash[i], signature[i])
    return signature.astype(int)

def lsh(signature, t):

    n = len(signature)

    # By Frasincar (2018), we use n = r*b for length of columns of signature matrix. Approx for the threshold is (1/b)^(1/r)
    r_best = 1
    b_best = 1
    best = 1
    for r in range(1, n + 1):
        for b in range(1, n + 1):
            if r * b == n:
                # Valid pair.
                approximation = (1 / b) ** (1 / r)
                if abs(approximation - t) < abs(best - t):
                    best = approximation
                    r_best = r
                    b_best = b

    candidates = np.zeros((len(signature[0]), len(signature[0])))
    for band in range(b_best):
        buckets = dict()
        start_row = r_best * band
        end_row = r_best * (band + 1)
        strings = ["".join(signature[start_row:end_row, column].astype(str)) for column in range(len(signature[0]))]
        ints = [int(string) for string in strings]
        hashes = [integer % sys.maxsize for integer in ints]

        # Add all item hashes to the correct bucket.
        for item in range(len(hashes)):
            hash_value = hashes[item]
            if hash_value in buckets:

                # All items already in this bucket are possible duplicates of this item.
                for candidate in buckets[hash_value]:
                    candidates[item, candidate] = 1
                    candidates[candidate, item] = 1
                buckets[hash_value].append(item)
            else:
                buckets[hash_value] = [item]
    return candidates.astype(int)

### start with the duplication method
  
  f = open("/Users/nahomtsehaie/Downloads/Computer science for BA paper/TVs-all-merged.json")
    data = json.load(f)

    std_data = standardize_data(data)

    #freq_words = common_words(std_data)
    #freq_words = freq_words[:15] # use 15 most frequent words for Binary Vector

    duplicates = duplicates_matrix(std_data)
    binary_vector = calc_bin_vector(std_data)

    n = round(round(0.5 * len(binary_vector)) / 100) * 100

    #Perform MinHashing
    signature_matrix = minhash(binary_vector, n)

 thresholds = [0.20, 0.25, 0.26, 0.27, 0.28, 0.30]

    comparisons_total = []
    frac_of_comp_total = []
    pq_total = []
    pc_total = []
    f1_star_total = []

    for t in thresholds:
        print("t = ", t)
        candidates = lsh(signature_matrix, t)
        comparisons = np.sum(candidates) / 2
        frac_of_comp = comparisons / comb(len(std_data), 2)

        correct = np.where(duplicates + candidates == 2, 1, 0)
        n_correct = np.sum(correct) / 2

        pq = n_correct / comparisons
        pc = n_correct / (np.sum(duplicates) / 2)

        f1_star = 2 * pq * pc / (pq + pc)

        comparisons_total.append(comparisons)
        frac_of_comp_total.append(frac_of_comp)
        pq_total = pq_total.append(pq)
        pc_total = pc_total.append(pc)
        f1_star_total = f1_star_total.append(f1_star)
        print(comparisons, frac_of_comp, pq, pc, f1_star)

    plt.figure(figsize=(20,10))
    plt.plot(thresholds, pq_total)
    plt.xlabel('Fraction comparisons')
    plt.ylabel('Pair Quality')
    plt.show()

    
