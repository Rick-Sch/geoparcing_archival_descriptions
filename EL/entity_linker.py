# -*- coding: utf-8 -*-

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sentence_transformers.util import semantic_search
import numpy as np
import pandas as pd

model = None

class EntityLinker():

  def __init__(self, kb_types=None, kb_df_embed_dict=None, TypesSubset=None, 
               threshold=.95, near_hit_threshold=0.7, score_function=util.dot_score, model=model,
               device='cpu', manual_links={}, wd_ids_as_int=True):
    """
    Entity Linker for linking description locations to Wikidata.
    
    
    kb_types : dict with key: NER tag, value: list with wikidata ids for corresponding key
    kb_df_embed_dict : dict with key: feature type (e.g. Label), value: dict with df and embed.
    TypesSubset : dict with key: feature type (e.g. Label), 
                                 value: 
                                    dict with key: ner tag, 
                                            value: dict with 'df' and 'embed'.
    threshold: Threshold to determine direct match.
    near_hit_threshold: Threshold to determine near match.
    score_function: Function to calculate distance between entity and candidate entities
    model: sentence encoder model object
    device: device for sentence encoder and encodings
    manual_links: quick fix for missing locations from database. e.g. {'Domkerk' : {'label' : 'Domkerk',
                                                                                    'wikidata_id' : 'Q936545'}}
    wd_ids_as_int: Convert wikidata ids to only integers (e.g. 'Q803' -> 803). This speeds up indexing and filtering
    """

    self.kb_types = kb_types
    self.kb_df_embed_dict = kb_df_embed_dict
    self.TypesSubset = TypesSubset
    self.threshold = threshold
    self.near_hit_threshold = near_hit_threshold
    self.score_function = score_function
    self.model = model
    self.device = device
    self.manual_links = manual_links
    self.wd_ids_as_int = wd_ids_as_int
    
    if self.TypesSubset == None:
      self.TypesSubset = self.CreateTypesSubset(convert_to_numbers=self.wd_ids_as_int)



  def __call__(self, description, ner_entities, threshold=None, 
               near_hit_threshold=None, manual_links=None, device=None):
    "Link entities from a description to wikidata items. Wapper around LinkEntities"
    if threshold != None:
      self.threshold = threshold
    if near_hit_threshold != None:
      self.near_hit_threshold = near_hit_threshold
    if manual_links != None:
      self.manual_links.update(manual_links)
    if device != None:
      self.device = device
     
    return self.LinkEntities(description, ner_entities)



  def CreateTypesSubset(self, convert_to_numbers=True):
    'Create types subsets. Needs self.kb_types and self.kb_df_embed_dict'
    
    TYPES_SUBSETS = {}

    # Iterate over kb features (label, altlabel description)
    for entityFeature, df_embed_dict in self.kb_df_embed_dict.items():

      # Get df and embed for feature subset
      feature_df, feature_embed = df_embed_dict['df'], df_embed_dict['embed']
      # Make dict for entityFeature.
      TYPES_SUBSETS[entityFeature] = dict()

      # Iterate over entityTypes subsets, and create subsets for types
      for ent_type, ent_ids in self.kb_types.items():
        filter = feature_df['entityType'].isin(ent_ids)

        df_sub, embed_sub = self.Subset(feature_df, feature_embed, filter, 'entity' + entityFeature, initial_subsets=True)
        if convert_to_numbers == True:
            for column in df_sub.columns:
                if column not in ['entityLabel', 'entityAltLabel', 'entityDescription', 'entityAddress']:
                    df_sub[column] = df_sub[column].apply(lambda x : int(x[1:]) if type(x) == str else x)
        # Save subset in dict
        TYPES_SUBSETS[entityFeature][ent_type] = {'df' : df_sub,
                                                  'embed' : embed_sub
                                                  }
      
      
      # Also add all items for LOC tag
      filter = [True] * len(feature_df)

      df_sub, embed_sub = self.Subset(feature_df, feature_embed, filter, 'entity' + entityFeature, initial_subsets=True)
      # Save subset in dict
      TYPES_SUBSETS[entityFeature]['LOC'] = {'df' : df_sub,
                                                  'embed' : embed_sub
                                                  }
    return TYPES_SUBSETS


  def Subset(self, df, embeddings, filter, entityFeature='entityLabel', 
             initial_subsets=False):
    'Subset df and embeddings with (boolean) filter'

    # Create df subset, with only unique wikidata ids.
    # self.dfSubset = df[filter].reset_index(drop=True).groupby(['entity', entityFeature]).first()
    if entityFeature.startswith('entity') == False:
      entityFeature = 'entity' + entityFeature

    self.embeddings = embeddings
    self.filter = filter
    # Only keep unique Qs 
    if initial_subsets == True:
      self.dfSubset = df[filter].reset_index(drop=True).reset_index().groupby(['entity', entityFeature]).first()
      self.embedSubset = embeddings[np.array(filter)][self.dfSubset['index'].values]
    
    else:
      self.dfSubset = df[filter].reset_index(drop=True).reset_index().groupby(['entity', entityFeature]).first()
      use_column = 'index'
      # if 'level_0' in self.dfSubset.index:
      #   use_column = 'level_0'
      try:
        self.embedSubset = embeddings[np.array(self.filter)][self.dfSubset['level_0'].values]
      except:
        self.embedSubset = embeddings[self.dfSubset[use_column].values]

    # Reset index again (so indexes line up with embeddings array)
    self.dfSubset.reset_index(inplace=True)

    return self.dfSubset, self.embedSubset

  

  def LinkEntities(self, description, ner_entities):
    "Link entities from a description to wikidata items"

    self.description_text, self.ner_entities = description, ner_entities

    # Create Linked dict for results
    self.Linked = {}

    # Create dict with entity_type : [entities]
    self.entity_dict = dict()

    # iterate over entities
    for ner_ent in set(self.ner_entities):
      name, entity_tag = ner_ent
      if name in self.manual_links.keys():
        self.Linked[name] = self.manual_links[name]
        continue

      if entity_tag not in self.entity_dict.keys():
          self.entity_dict[entity_tag] = [name]
      else:
        self.entity_dict[entity_tag].append(name)
        
    # List with all entity types    
    entity_types = list(self.entity_dict.keys())
        
    # Get city subset (and link them). Since some cities get LOC tag, add this
    # to city list.
    city_locations = []
    if 'CITY' in entity_types:
        city_locations += self.entity_dict['CITY']
    if len(city_locations) == 0 and 'LOC' in entity_types:
        city_locations += self.entity_dict['LOC']
    
    self.cities = [803]
    if len(city_locations) > 0:
      # Link City
      self.cities += self.FindEntity(city_locations, 'CITY')
    
    # Link other in order
    possible_entity_types = ['STREET', 'LOC', 'NEIGH', 'RIVER', 'WATER']
    self.present_entities = [i for i in possible_entity_types if i in entity_types]

    for ent_type in self.present_entities:
      hit_results = self.FindEntity(self.entity_dict[ent_type], ent_type, cities=self.cities)

    return self.Linked


  def FindEntity(self, entities, ent_type, cities=[]):
    
    # Find initial hits
    hit_results = self.SearchDB(entities, ent_type, cities, subset='Label')
    
    hits = []
    # Iterate over hits, keep valid hits.
    for entity, (valid, hit_df) in zip(entities, hit_results):
      # If true hit, add hit df to hits list
      if valid == 'HIT':
        hits.append(hit_df)
        self.Linked[entity] = {'label' : hit_df.entityLabel.tolist(),
                               'wikidata_id' : hit_df.entity.tolist()}

      if valid == 'NEAR_HIT':
        # Lookup best match for sent part
        merged_df = self.sent_match(entity, self.description_text, hit_df, self.df, self.embed, window=30)

        # If results, take upper item (best text match)
        if len(merged_df) != 0:
          self.Linked[entity] = {'label' : merged_df.entityLabel.tolist()[:1],
                                 'wikidata_id' : merged_df.entity.tolist()[:1]
                                 }
          hits.append(merged_df)

        else:
          # print(hit_df)
          try:
            hit_df = self.choose(entity=entity,
                                entity_ids=hit_df.entity.tolist(), 
                                subset='Description')
            
          except:
            try:
              hit_df = self.choose(entity=entity,
                                  entity_ids=hit_df.entity.tolist(), 
                                  subset='AltLabel')
            except:
              self.Linked[entity] = {'label' : False,
                                    'wikidata_id' : False
                                    }
              continue

          hits.append(hit_df)

          self.Linked[hit_df.name[0]] = {'label' : hit_df.entityLabel.tolist(),
                                        'wikidata_id' : hit_df.entity.tolist()}

      if valid == False:
        self.Linked[entity] = {'label' : False,
                               'wikidata_id' : False
                               }
        
    # If type is city, return list of Q's
    if ent_type == 'CITY':
      if len(hits) > 0:
        hits_df_stack = pd.concat(hits)
        hits = hits_df_stack.entity.tolist()
      else:
        hits = []

    return hits



  def sent_match(self, entity, description_text, hit_df, kb_df, embed, window=30):
    'Match best candidate for sentence part of size window (or start) + entity + window (or end) containing the entity'
    
    # Create window first
    ent_len, txt_len = len(entity), len(description_text)
    ent_strt = description_text.index(entity)
    ent_end = ent_strt + ent_len
    
    # Define start index
    end = ent_end + window
    if end > txt_len:
      end = txt_len

    # Define end index
    strt = ent_strt - window
    if strt < 0:
      strt = 0

    # get sentence part
    self.sent = description_text[strt:end]
    
    # encode sent
    query = self.model.encode([self.sent], normalize_embeddings=True, device=self.device)
    sent_hits = util.semantic_search(query, embed,
                                score_function=self.score_function, 
                                top_k=30)
    
    # create dataframe with hits
    self.sent_hit_df = self.hits_to_df(kb_df, sent_hits, [entity])[0]

    try:
      # Merge sent_hit_df with hits_df
      self.merged_df = hit_df.merge(self.sent_hit_df[['entity', 'score']], on='entity', suffixes=('_lab', '_txt')).sort_values('score_txt', ascending=False)
    except:
      return hit_df

    return self.merged_df



  
  def choose(self, entity, entity_ids, subset='AltLabel'):

    self.choose_results = self.SearchDB([entity], subset=subset, compare_ids=entity_ids)
    
    hit_concat = pd.concat([hit[1] for hit in self.choose_results if hit[0] !=False]).sort_values('score', ascending=False)
    
    hit_df = hit_concat[hit_concat.score == hit_concat.score.max()]

    return hit_df
    


 # OPTIE: Matchen op zowel label als altlabel. Dat gebruiken om te bepalen welke 
 # entity gekozen moet worden als meerdere opties boven 0.8
 
  def SearchDB(self, entities, ent_type='LOC', cities=[], subset='Label', compare_ids=[]):
    'Find best match for entity with ent_type'

    if ent_type not in self.kb_types.keys():
      ent_type = 'LOC'

    # If compare ids, create ids subset
    if len(compare_ids) > 0:
      # print(entities, compare_ids)
      # Look up in LOC because all items are in there
      self.df, self.embed = self.TypesSubset[subset]['LOC']['df'], self.TypesSubset[subset]['LOC']['embed']
      
      # Get df and embedding for entity ids
      self.df, self.embed = self.Subset(self.df, self.embed, self.df.entity.isin(compare_ids), entityFeature=subset)
      
    # If normal search, use subset for type (and/or city)
    else:
      self.df, self.embed = self.TypesSubset[subset][ent_type]['df'], self.TypesSubset[subset][ent_type]['embed']
      
      if len(cities) > 0:
        self.df, self.embed = self.Subset(self.df, self.embed, (self.df.entityCity.isin(cities) | self.df.entityLocation.isin(cities)))

    # Encode entities
    self.query = self.model.encode(entities, normalize_embeddings=True, device=self.device)
    # Find hits
    self.hits = util.semantic_search(self.query, self.embed,
                                score_function=self.score_function, 
                                top_k=30)
    # print(self.hits)
    # Make DataFrame for each hit

    self._hits_df = self.hits_to_df(self.df, self.hits, entities)

    # Determine hit
    self._hit_results = self.determine_hit(self._hits_df)

    return self._hit_results
  
  

  def determine_hit(self, hits_df):
    
    results = []
    for df in hits_df:
      HIT = False  # Hit flag
      max_score = df.score.iloc[0]
      if max_score < self.near_hit_threshold:
        # print(df)
        results.append((HIT, HIT))
        continue

      if max_score >= self.threshold:
        HIT = 'HIT'
        max_score_hits = df[df['score'] == max_score]
        results.append((HIT, max_score_hits))
        continue

      if max_score >= self.near_hit_threshold:
        HIT = 'NEAR_HIT'
        max_score_hits = df[df['score'] >= self.near_hit_threshold]
        results.append((HIT, max_score_hits))
        continue
      
    return results



  def hits_to_df(self, df, hits, queries):
    'List with query hits (one list per query result) to list with dataframes'
    
    # List to store dfs
    hits_dfs = []
    # Iterate over hit lists
    for name, query_hits in zip(queries, hits):
      # Create DataFrame
      try:
        hit_df = pd.DataFrame(query_hits).sort_values('score', ascending=False)
        hit_df.score = hit_df.score.round(4)
      except:
        hit_df = pd.DataFrame([{'score' : 0, 'corpus_id' : False}])
        hits_dfs.append(hit_df)
        continue
      # Add name column
      hit_df['name'] = name
      # Add wikidata ids to df
      link_label = df.loc[hit_df.corpus_id, ['entity', 'entityLabel', 'entityType']].reset_index(drop=True)
      hit_df = hit_df.join(link_label)
      # Append df to list
      hits_dfs.append(hit_df)
      
    # # Concat dfs and group by name
    # hits_dfs = pd.concat(hits_dfs).groupby('name')
    return hits_dfs


  def hits_to_wikidata(self, df, hits, queries):
    'List with query hits (one list per query result) to dict with query: [wikidata_ids]'
    wikidata_hits = dict()

    for query, query_hits in zip(queries, hits):
      wikidata_hits[query] = [df.iloc[hit['corpus_id']]['entity'] for hit in query_hits]
    
    return wikidata_hits


  def print_hits(self, hits, df, entities=None):
    'Print hits'
    if entities != None:
      hits = zip(entities, hits)
    else:
      hits = zip([None for i in hits], hits)

    for name, hits_dicts in hits:
      print("\nNAME:", name,  '\n')
      for hit in hits_dicts:
        row = df.iloc[hit['corpus_id']]
        print('{} - {} ({})'.format(row.entityLabel, round(hit['score'], 4), row.entity))