import values

def concept_picker(competence_dict):

    """
    Sample JSON input:

    cluster_pool_competences = {
    "Cluster1": {"concept1": 0.2, "concept2": 0.3},
    "Cluster2": {"concept3": 0.7, "concept4": 0.9},
    }
    """

    upper_threshold = values.UPPER_THRESHOLD
    lower_threshold = values.LOWER_THRESHOLD

    # Removes concept clusters which fall outside the threshold range
    for cluster, rating_dict in competence_dict.items():
        for concept, rating_val in rating_dict.items():
            if rating_val > upper_threshold or rating_val < lower_threshold:
                del competence_dict[cluster][concept]

    # final list of concepts
    sorted_comptence_scores = sorted(competence_dict.items(), key=lambda x: x[1])

    desired_concepts = [
        concept
        for cluster, concepts_dict in sorted_comptence_scores
        for concept in concepts_dict.keys()
    ]

    return desired_concepts







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

    # sorted_concept_weights =
    # dict(sorted(concept_weights.items(), key=lambda item: item[0]))
