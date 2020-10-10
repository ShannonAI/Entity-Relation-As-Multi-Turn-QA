import json


ace2004_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2004_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2004_relations = ['ART', 'EMP-ORG', 'GPE-AFF', 'OTHER-AFF', 'PER-SOC', 'PHYS']
ace2004_relations_full = ['artifact','employment, membership or subsidiary','geo political affiliation','person or organization affiliation','personal or social','physical']

ace2005_entities = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']
ace2005_entities_full = ["facility","geo political","location","organization","person","vehicle","weapon"]
ace2005_relations = ['ART', 'GEN-AFF', 'ORG-AFF', 'PART-WHOLE', 'PER-SOC', 'PHYS']
ace2005_relations_full = ["artifact","gen affilliation",'organization affiliation','part whole','person social','physical']


if __name__=="__main__":
    #这里的模板应该保证我们的模型的query和对应的entity type或者(head_entity relation_type, end_entity_type)存在一一对应关系
    templates = {"qa_turn1":{},"qa_turn2":{}}
    for ent1,ent1f in zip(ace2005_entities,ace2005_entities_full):
        templates['qa_turn1'][ent1]="find all {} entities  in the context.".format(ent1f)
        for rel,relf in zip(ace2005_relations,ace2005_relations_full):
            for ent2,ent2f in zip(ace2005_entities,ace2005_entities_full):
                templates['qa_turn2'][str((ent1,rel,ent2))]="find all {} entities in the context that have {} {} relationship with {} entity XXX.".format(ent2f,'an' if relf[0] in ['a','o'] else 'a',relf,ent1f)
    with open("ace2005.json",'w') as f:
        json.dump(templates,f)
    for ent1,ent1f in zip(ace2004_entities,ace2004_entities_full):
        templates['qa_turn1'][ent1]="find all {} entities  in the context.".format(ent1f)
        for rel,relf in zip(ace2004_relations,ace2004_relations_full):
            for ent2,ent2f in zip(ace2004_entities,ace2004_entities_full):
                templates['qa_turn2'][str((ent1,rel,ent2))]="find all {} entities in the context that have {} {} relationship with {} entity XXX.".format(ent2f,'an' if relf[0] in ['a','o'] else 'a',relf,ent1f)
    with open("ace2004.json",'w') as f:
        json.dump(templates,f)