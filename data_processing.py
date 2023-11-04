import numpy as np
import pandas as pd
import pickle
import matplotlib as plt
from itertools import combinations
import re
import csv
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    print('Pre_processing data main')

    codes = {}
    dir_path = './data_raw/'

    # from what ive found so far the place that they got their drug gene interactions has changed the name from drug-gen to chem-gene
    # we will probably want to note that in the paper

    #Load all TSV's
    #chem-gene
    chem_gene_headers=["ChemicalName","ChemicalID","CasRN","GeneSymbol","GeneID","GeneForms","Organism","OrganismID","Interaction","InteractionActions","PubMedIDs"] #had to look into the file for these
    chem_gene_interaction = pd.read_csv(dir_path+'CTD_chem_gene_ixns.tsv', sep='\t', comment='#', names=chem_gene_headers)
    # print(chem_gene_interaction.head())

    codes['drugname2mesh']={row[0].upper():row[1] for idx, row in chem_gene_interaction[['ChemicalName','ChemicalID']].drop_duplicates().iterrows()}
    codes['mesh2drugname']={row[0].upper():row[1] for idx, row in chem_gene_interaction[['ChemicalID','ChemicalName']].drop_duplicates().iterrows()}

    print('chem_gene line count',len(chem_gene_interaction.index))
    #cleaning out the duplicates and setting gene and chem names

    chem_gene_interaction = chem_gene_interaction[['GeneID', 'ChemicalName']].drop_duplicates()
    chem_gene_interaction['ChemicalName'] = chem_gene_interaction['ChemicalName'].apply(lambda x: x.upper() if type(x) == str else x)

    chem_gene_interaction['GeneID'] = chem_gene_interaction['GeneID'].apply(lambda x: 'gene_' + str(x))
    chem_gene_interaction['ChemicalName'] = chem_gene_interaction['ChemicalName'].apply(lambda x: 'drug_' + x)

    chem_gene_interaction.drop_duplicates(inplace=True)
    print('chem_gene line count',len(chem_gene_interaction.index))

    #pathways
    pathways = pd.read_csv(dir_path + 'CTD_D000086382_pathways_20231101180127.tsv', sep='\t')
    path_sim = pd.concat([pd.DataFrame(list(combinations(pathway, 2, )), columns=['gene1', 'gene2']) for pathway in
                          pathways['Association inferred via'].apply(
                              lambda x: x.split('|') if '|' in x else None).dropna().values]).drop_duplicates()
    # print(pathways.head())

    #load processes pairwise genes from KEGG
    path_sim_kegg = pd.read_csv(dir_path + 'KegglinkevaluationPPPN_1.txt', header=None, sep='\t')
    # print(path_sim_kegg.head())
    # renaming the col from 0 and 1  to gene1 and gene 2
    path_sim_kegg.columns = ['gene1', 'gene2', 'positive']
    path_sim_kegg.replace('PP', 1, inplace=True)
    path_sim_kegg.replace('PN', 0, inplace=True)

    path_sim_kegg = path_sim_kegg.loc[path_sim_kegg['positive'] == 1, ['gene1', 'gene2']]

    gene_name = pd.read_excel(dir_path + 'All_Human_Protein_Coding_Genes_3_27_2020.xlsx')
    gene_dict = {row['Gene Id']: row['Gene Symbol'] for _, row in gene_name[['Gene Id', 'Gene Symbol']].iterrows()}
    #had to make this since they forgot to?
    codes['gene_symbol2id']={row['Gene Symbol']: row['Gene Id'] for _, row in gene_name[['Gene Id', 'Gene Symbol']].iterrows()}
    path_sim_kegg.dropna()

    # map the gene names to their id number
    path_sim_kegg['gene1'] = path_sim_kegg['gene1'].apply(lambda x: gene_dict.get(x))
    path_sim_kegg['gene2'] = path_sim_kegg['gene2'].apply(lambda x: gene_dict.get(x))
    path_sim_kegg.dropna(inplace=True)

    #PHARMAKD pathways
    pathway_ace_inhibitor = list(
        {'ATP6AP2', 'MAPK1', 'AGTR2', 'ATP6AP2', 'REN', 'MAS1', 'TGFB1', 'MAPK3', 'ATP6AP2', 'MAPK3', 'AGTR1', 'TGFB1',
         'MAPK1', 'NOS3', 'BDKRB2', 'BDKRB2', 'BDKRB1', 'NR3C2', 'CYP11B2', 'AGTR1', 'CYP11B2', 'AGTR1', 'AGT', 'KNG1',
         'CYP11B2', 'ACE'})
    pathway_fluv = ['CYP1A2', 'CYP2C19', 'CYP3A']
    pathway_losartan = list(
        {'AGTR1', 'CYP2C9', "CYP3A4", 'CYP2C9', "CYP3A4", 'CYP2C9', "CYP3A4", 'CYP2C9', "CYP3A4", 'UGT1A1', "UGT2B7"})

    path_sim=pd.concat([path_sim]+[path_sim_kegg]+[pd.DataFrame(list(combinations(pathway,2,)),columns=['gene1','gene2']) for pathway in [pathway_ace_inhibitor,pathway_fluv,pathway_losartan]])
    path_sim['gene1']=path_sim['gene1'].apply(lambda x: codes['gene_symbol2id'].get(x))
    path_sim['gene2']=path_sim['gene2'].apply(lambda x: codes['gene_symbol2id'].get(x))
    path_sim.dropna(inplace=True)
    path_sim['gene1']=path_sim['gene1'].apply(lambda x: 'gene_'+str(int(x)))
    path_sim['gene2']=path_sim['gene2'].apply(lambda x: 'gene_'+str(int(x)))
    path_sim.drop_duplicates(inplace=True)
    print('path_sim length: ',len(path_sim))

    # phenotypes

    phenotypes = pd.read_csv(dir_path + 'CTD_D000086382_diseases_20231101180316.tsv', sep='\t')
    phenotypes.head()

    codes['phenotype_id_to_name'] = {row[0]: row[1] for idx, row in phenotypes[
        ['Phenotype Term ID', 'Phenotype Term Name']].drop_duplicates().iterrows()}
    drug_phenotype = phenotypes['Chemical Inference Network'].dropna().apply(lambda x: x.split('|')).apply(
        pd.Series).merge(phenotypes['Phenotype Term ID'], left_index=True, right_index=True).melt(
        id_vars=['Phenotype Term ID'], value_name='drug').drop('variable', axis=1).dropna()
    drug_phenotype['drug'] = drug_phenotype['drug'].apply(lambda x: x.upper())
    drug_phenotype.dropna(inplace=True)

    drug_phenotype['Phenotype Term ID'] = drug_phenotype['Phenotype Term ID'].apply(lambda x: 'phenotype_' + x)
    drug_phenotype['drug'] = drug_phenotype['drug'].apply(lambda x: 'drug_' + x)
    drug_phenotype = drug_phenotype[['drug', 'Phenotype Term ID']]

    gene_phenotype = phenotypes['Gene Inference Network'].dropna().apply(lambda x: x.split('|')).apply(pd.Series).merge(
        phenotypes['Phenotype Term ID'], left_index=True, right_index=True).melt(id_vars=['Phenotype Term ID'],
                                                                                 value_name='gene').drop('variable',
                                                                                                         axis=1).dropna()
    gene_phenotype['Phenotype Term ID'] = gene_phenotype['Phenotype Term ID'].apply(lambda x: 'phenotype_' + x)

    gene_phenotype['gene'] = gene_phenotype['gene'].apply(lambda x: codes['gene_symbol2id'].get(x))
    gene_phenotype.dropna(inplace=True)
    gene_phenotype['gene'] = gene_phenotype['gene'].apply(lambda x: 'gene_' + str(int(x)))
    gene_phenotype = gene_phenotype[['gene', 'Phenotype Term ID']]

    #checking lens
    print(len(set(chem_gene_interaction['ChemicalName'].values).intersection(set(drug_phenotype['drug'].values))))
    print(len(set(drug_phenotype['drug'].values)))

    #bait and prey
    baits_prey = pd.read_excel(dir_path + 'bait_and_prey.xlsx')
    # print(baits_prey.head())
    baits_prey = baits_prey[['Bait', 'PreyGene']]
    print('unique baits, unique prey genes',baits_prey['Bait'].nunique(), baits_prey['PreyGene'].nunique())

    baits_prey['Bait'] = baits_prey['Bait'].apply(lambda x: 'bait_' + x)
    baits_prey['PreyGene'] = baits_prey['PreyGene'].apply(lambda x: codes['gene_symbol2id'].get(x))
    baits_prey.dropna(inplace=True)
    baits_prey['PreyGene'] = baits_prey['PreyGene'].apply(lambda x: 'gene_' + str(int(x)))

    # size of drug target, pathway, host gene, phenotype-related genes
    print('size of drug target, pathway, host gene, phenotype-related genes',chem_gene_interaction['GeneID'].nunique(), len(set(path_sim[['gene1', 'gene2']].values.ravel())), len(set(baits_prey['PreyGene'].unique())),gene_phenotype['gene'].nunique())

    # number of intersection between host genes and drug target
    print('number of intersection between host genes and drug target',len(set(baits_prey['PreyGene'].unique()).intersection(chem_gene_interaction['GeneID'].unique())))

    # intersection between target and pathways
    print('intersection between target and pathways',len(set(chem_gene_interaction['GeneID'].unique()).intersection(set(path_sim[['gene1', 'gene2']].values.ravel()))))

    # intersection between pathways and host gene
    print('intersection between pathways and host gene',len(set(baits_prey['PreyGene'].unique()).intersection(set(path_sim[['gene1', 'gene2']].values.ravel()))))

    # intersection between pathways, target genes, host gene
    print('intersection between pathways, target genes, host gene',len(set(baits_prey['PreyGene'].unique()).intersection(set(path_sim[['gene1', 'gene2']].values.ravel())).intersection(set(chem_gene_interaction['GeneID'].unique()))))



    #item to inx mapping
    chem_gene_interaction.columns = ['node1', 'node2']  # gene, drug
    path_sim.columns = ['node1', 'node2']  # gene1, gene2
    baits_prey.columns = ['node1', 'node2']  # bait, preygene
    gene_phenotype.columns = ['node1', 'node2']  # gene, phenotype
    drug_phenotype.columns = ['node1', 'node2']  # drug, phenotye

    chem_gene_interaction['type'] = 'gene-drug'
    path_sim['type'] = 'gene-gene'
    baits_prey['type'] = 'bait-gene'
    gene_phenotype['type'] = 'gene-phenotype'
    drug_phenotype['type'] = 'drug-phenotype'

    edge_index = pd.concat([chem_gene_interaction, path_sim, baits_prey, gene_phenotype, drug_phenotype])
    edge_index['node1'] = edge_index['node1'].astype(str)
    edge_index['node2'] = edge_index['node2'].astype(str)


    #label encoders
    le = LabelEncoder()
    le.fit(np.concatenate((edge_index['node1'], edge_index['node2'])))

    edge_index['node1'] = le.transform(edge_index['node1'])
    edge_index['node2'] = le.transform(edge_index['node2'])

    print(len(le.classes_))

    #pre trained embeddings
    entity_emb=np.load(dir_path+'DRKG_TransE_l2_entity.npy')
    emb_size=entity_emb.shape[1]

    entity_idmap_file = dir_path + 'entities.tsv'
    relation_idmap_file = dir_path + 'relations.tsv'

    baits_drkg = ['Disease::' + entity.split('_')[1] for entity in le.classes_ if entity.split('_')[0] == 'bait']
    gene_drkg = ['Gene::' + entity.split('_')[1] for entity in le.classes_ if entity.split('_')[0] == 'gene']
    phenotype_drkg = ['Biological Process::' + entity.split('_')[1] for entity in le.classes_ if
                      entity.split('_')[0] == 'phenotype']

    drugname2external = pd.concat(
        [pd.read_csv(dir_path + 'alldrugbank.csv', index_col=0).rename(columns={'drugbank_id': 'id'}),
         pd.read_csv(dir_path + 'nondrugbank.csv', index_col=0).rename(columns={'chembl': 'id'})]).groupby(
        'drugname', as_index=False).first()
    drugname2id = {row[0].upper(): row[1] for _, row in drugname2external[['drugname', 'id']].iterrows()}
    drug_drkg = ['Compound::' + drugname2id.get(entity.split('_')[1], '') for entity in le.classes_ if
                 entity.split('_')[0] == 'drug']

    entity_map = {}
    entity_id_map = {}
    relation_map = {}
    with open(entity_idmap_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name', 'id'])
        for row_val in reader:
            entity_map[row_val['name']] = int(row_val['id'])
            entity_id_map[int(row_val['id'])] = row_val['name']

    with open(relation_idmap_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name', 'id'])
        for row_val in reader:
            relation_map[row_val['name']] = int(row_val['id'])

    # handle the ID mapping
    bait_ids = []
    gene_ids = []
    drug_ids = []
    phenotype_ids = []

    for bait in baits_drkg:
        bait_ids.append(entity_map.get(bait))

    for gene in gene_drkg:
        gene_ids.append(entity_map.get(gene))

    for drug in drug_drkg:
        drug_ids.append(entity_map.get(drug))

    for phenotype in phenotype_drkg:
        phenotype_ids.append(entity_map.get(phenotype))

    bait_emb = np.array([entity_emb[bait_id] if bait_id is not None else np.zeros(emb_size) for bait_id in bait_ids])
    drug_emb = np.array([entity_emb[drug_id] if drug_id is not None else np.zeros(emb_size) for drug_id in drug_ids])
    gene_emb = np.array([entity_emb[gene_id] if gene_id is not None else np.zeros(emb_size) for gene_id in gene_ids])
    phenotype_emb = np.array(
        [entity_emb[phenotype_id] if phenotype_id is not None else np.zeros(emb_size) for phenotype_id in
         phenotype_ids])
    # How many missing in drugs?
    print('How many missing in drugs?',len(drug_ids),len([gene_id for gene_id in drug_ids if gene_id is not None]))

    # How many missing in genes?
    print('How many missing in genes?',len(gene_ids), len([gene_id for gene_id in gene_ids if gene_id is not None]))

    # How many missing in phenotypes?
    print('How many missing in phenotypes?',len(phenotype_ids), len([gene_id for gene_id in phenotype_ids if gene_id is not None]))

    node_features = np.concatenate((bait_emb, drug_emb, gene_emb, phenotype_emb))

    exp_id = 'v0'
    processed_dir = './data_processed/'
    #save to pickle
    edge_index.to_pickle(processed_dir + 'edge_index_' + exp_id + '.pkl')
    pickle.dump(le, open(processed_dir + 'LabelEncoder_' + exp_id + '.pkl', 'wb'))
    pickle.dump(node_features, open(processed_dir + 'node_feature_' + exp_id + '.pkl', 'wb'))
    pickle.dump(codes, open(processed_dir + 'codes_' + exp_id + '.pkl', 'wb'))



