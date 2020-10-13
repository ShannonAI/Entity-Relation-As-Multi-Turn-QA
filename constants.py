import json

question_templates_path = "./data/query_templates/"
ace2004_question_templates = json.load(
    open(question_templates_path+'ace2004.json'))
ace2005_question_templates = json.load(
    open(question_templates_path+'ace2005.json'))

tag_idxs = {'B': 0, 'M': 1, 'E': 2, 'S': 3, 'O': 4}

ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_entities_full = ["facility", "geo political",
                         "location", "organization", "person", "vehicle", "weapon"]
ace2004_relations = ['ART', 'EMP-ORG',
                     'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']
ace2004_relations_full = ['artifact', 'employment, membership or subsidiary',
                          'geo political affiliation', 'person or organization affiliation', 'personal or social', 'physical']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility", "geo political",
                         "location", "organization", "person", "vehicle", "weapon"]
ace2005_relations = ['ART', 'GEN-AFF',
                     'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact", "gen affilliation",
                          'organization affiliation', 'part whole', 'person social', 'physical']

# index of ace2004 and ace2005 frequency matrix
ace2004_idx1 = {'FAC': 0, 'GPE': 1, 'LOC': 2,
                'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6}
ace2005_idx1 = {'FAC': 0, 'GPE': 1, 'LOC': 2,
                'ORG': 3, 'PER': 4, 'VEH': 5, 'WEA': 6}
ace2005_idx2t = {}
for i, rel in enumerate(ace2005_relations):
    for j, ent in enumerate(ace2005_entities):
        ace2005_idx2t[(rel, ent)] = i*len(ace2005_relations)+j+i
ace2005_idx2 = ace2005_idx2t
ace2004_idx2t = {}
for i, rel in enumerate(ace2004_relations):
    for j, ent in enumerate(ace2004_entities):
        ace2004_idx2t[(rel, ent)] = i*len(ace2004_relations)+j+i
ace2004_idx2 = ace2004_idx2t
# statistics on the training set
ace2005_dist = [[0,   0,   0,   0,   0,   0,   0,   0,   3,   1,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,  33, 116,  39,   2,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,  10,  22,  11,   0,
                 0,   0,   0],
                [30,   0,   0,   0,   0,  60,  61,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,  19,   0,   0,   0,   1, 143,  47,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   4,  14,   9,   0,
                 0,   0,   0],
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0, 120,  31,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   3,  18,   8,   0,
                 0,   0,   0],
                [35,   0,   0,   0,   0,  35,  10,   0, 149,  20,   0,   0,   0,
                 0,   0,   5,   0,  12,   0,   0,   0,   0, 147,   1,  81,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   3,   5,   0,   0,
                 0,   0,   0],
                [67,   1,   0,   2,   0, 113,  77,   0, 270,  27,  10,  32,   0,
                 0,   0, 587,   0, 844,   5,   0,   0,   0,   0,   0,   4,   0,
                 0,   0,   0,   0,   0,   4, 434,   0,   0, 281, 494, 213,   4,
                 0,   1,   0],
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 8,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0],
                [0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                 0,   0,   0]]
# statistics on the entire data set
ace2004_dist = [[0,    1,    0,    0,    0,    0,    0,    0,    2,    0,    0,
                 0,    0,    0,    0,    1,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    1,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,   31,  104,   28,    0,    0,    0,    0],
                [8,    1,    1,    0,    0,   27,    8,    0,    8,    0,    3,
                 0,    0,    0,    1,   12,    5,    0,    0,    0,    0,    0,
                 1,    0,    3,   20,    0,    0,    0,    0,    0,    0,    1,
                 0,    0,    0,  236,   55,    0,    0,    0,    0],
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    1,
                 0,    0,    0,    0,    5,    0,    0,    0,    0,    0,    0,
                 1,    0,    1,    1,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,   65,   38,    0,    0,    0,    0],
                [28,    0,    0,    0,    0,   24,    2,    0,  132,    0,  120,
                 1,    0,    0,    0,  203,   16,    0,    0,    0,    0,    0,
                 4,    1,    6,    6,    0,    0,    0,    0,    0,    0,    1,
                 0,    0,    4,   19,    2,    1,    0,    0,    0],
                [55,    0,    0,    0,    0,   29,   25,    3,  311,    1, 1035,
                 8,    0,    0,    0,  276,    9,    0,    1,    0,    0,    1,
                 5,    4,   18,   69,    0,    0,    0,    0,    0,    0,  363,
                 0,    0,  168,  328,   55,    8,    9,   16,    0],
                [0,    3,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    8,   19,    5,    0,    0,    3,    0],
                [0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
                 0,    0,    3,    1,    3,    0,    0,    2,    3]]
