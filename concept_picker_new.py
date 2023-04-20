import values
import json
import numpy as np


def concept_picker(competence_dict):

    """
    Sample Input(JSON):

    cluster_pool_competence_confidence = {
    "Cluster1": {"concept1": (0.2, 0.25), "concept2": (0.3, 0.95)},
    "Cluster2": {"concept3": (0.4, 0.56), "concept4": (0.5, 0.78)},
    "Cluster3": {"concept5": (0.9, 0.24), "concept6": (0.9, 0.3)},
    "Cluster4": {
        "concept7": (0.7, 0.34),
        "concept8": (0.1, 0.34),
        "concept9": (0.25, 0.72),
    },
    }

    tuple: (x, y) -> (competence_rating, confidence_value)

    Sample Output(JSON):

    ["concept4", "concept6", "concept7", "concept9"]
    """

    upper_threshold = values.UPPER_THRESHOLD
    lower_threshold = values.LOWER_THRESHOLD

    # Removes concept clusters which fall outside the threshold range
    new_cluster_pool = {}
    for cluster, rating_dict in competence_dict.items():
        new_concepts_dict = {}
        for concept, (competence, confidence) in rating_dict.items():
            if confidence > upper_threshold or confidence < lower_threshold:
                continue
            new_concepts_dict[concept] = (competence, confidence)
        if new_concepts_dict:
            new_cluster_pool[cluster] = new_concepts_dict

    competence_dict = new_cluster_pool

    # Aggregates eacch cluster using harmonic mean
    cluster_harmonic_means = {}

    for cluster, concepts_dict in competence_dict.items():
        confidences = []
        for _, conf in concepts_dict.values():
            conf_reciprocal = 1 / conf
            confidences.append(conf_reciprocal)
        harmonic_mean = len(confidences) / sum(confidences)
        cluster_harmonic_means[cluster] = harmonic_mean

    # Extracts concepts from clusters which fall below 50% confidence
    low_confidence_concepts = [
        concept
        for cluster in cluster_harmonic_means
        if cluster_harmonic_means[cluster] < values.CONFIDENCE
        for concept in competence_dict[cluster]
    ]

    # parses list comprehension into JSON object
    json_object = json.dumps(low_confidence_concepts)

    return json_object
