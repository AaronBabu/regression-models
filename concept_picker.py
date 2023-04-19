import values
import json


def concept_picker(competence_dict):

    """
    Sample Input(JSON):

    cluster_pool_competence_confidence = {
    "Cluster1": {"concept1": (0.2, 0.1), "concept2": (0.3, 0.95)},
    "Cluster2": {"concept3": (0.4, 0.15), "concept4": (0.5, 0.2)},
    "Cluster3": {"concept5": (0.9, 0.9), "concept6": (0.9, 0.3)},
    "Cluster4": {
        "concept7": (0.7, 0.45),
        "concept8": (0.1, 0.6),
        "concept9": (0.25, 0.34),
    },
    }

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

    # final list of concepts (below 50% confidence)
    low_confidence_concepts = [
        concept
        for cluster in competence_dict.values()
        for concept, (_, confidence) in cluster.items()
        if confidence <= values.CONFIDENCE
    ]

    json_object = json.dumps(low_confidence_concepts)

    return json_object


# IGNORE STUFF BELOW

# Bias for selecting concepts from remaining pool (determining weights)
# Weights concepts that were not asked at all or minimally asked
"""
concept_weights = {}
    for concept_name, response in concept_responses.items():
        if response["totalAnswered"] == 0:
            concept_weights[concept_name] = 1
        else:
            concept_weights[concept_name] = 1 / response["totalAnswered"]
"""


"""
sorted_comptence_scores = sorted(competence_dict.items(), key=lambda x: x[1])

desired_concepts = [
        concept
        for cluster, concepts_dict in sorted_comptence_scores
        for concept in concepts_dict.keys()
    ]

return desired_concepts
"""


# sorted_concept_weights =
# dict(sorted(concept_weights.items(), key=lambda item: item[0]))
