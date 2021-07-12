# -*- coding: utf-8 -*-

import itertools
# fuzz is used to compare TWO strings
from fuzzywuzzy import fuzz

# process is used to compare a string to MULTIPLE other strings
from fuzzywuzzy import process

class FuzzyEntityLinker():
    "Class for fuzzy entity linking."
    
    def __init__(self, kb, kb_types_map, threshold=85, fuzz_city=fuzz.ratio, fuzz_other=fuzz.partial_ratio):
        self.kb = kb
        self.kb_types_map = kb_types_map
        self.kb_types_subsets = self.get_type_subsets()
        self.threshold = threshold
        self.fuzz_city = fuzz_city
        self.fuzz_other = fuzz_other
        
    def LinkEntities(self, entities, text):
        'Link entities.'
        self.wd_lookup = dict() # Lookup these entities on wikidata
        self.linked_entities = dict()
        self.link_entities_manual = dict()
        self.city_matches = dict()
        
        
        # Create dict with entities
        self.entity_dict = dict()
        for entity in set(entities):
            name, entity_tag = entity
            if entity_tag not in self.entity_dict.keys():
                self.entity_dict[entity_tag] = [name]
            else:
                self.entity_dict[entity_tag].append(name)
                
        # List with all entity types    
        entity_types = list(self.entity_dict.keys())
        
        # Check for country:
        if 'COUNTRY' in entity_types:
            pass
            #CHECK COUNTRY
        
        # Get city subset (and link them). Since some cities get LOC tag, add this
        # to city list.
        city_locations = []
        if 'CITY' in entity_types:
            city_locations += self.entity_dict['CITY']
        if len(city_locations) == 0 and 'LOC' in entity_types:
            city_locations += self.entity_dict['LOC']
        # Update filtered subset with city if city_locations is not len zero
        if len(city_locations) >= 1:
            self.filtered_subset = self.get_city_and_subset(city_locations, text, scorer=self.fuzz_city)
        else:
            self.filtered_subset = self.kb
        
        # Link all other entities
        self.match_entities_in_subset(self.entity_dict, self.filtered_subset, text=text, scorer=self.fuzz_other)
        
        return self.linked_entities, self.link_entities_manual, self.wd_lookup

        
    def match_entities_in_subset(self, entity_dict, filtered_subset, text, scorer=fuzz.partial_ratio):
        
        
        possible_entity_types = ['STREET', 'LOC', 'NEIGH', 'RIVER', 'WATER']
        entities_ordered = [[(i, entity) for entity in entity_dict[i]] for i in possible_entity_types if i in entity_dict.keys()]
        entities_ordered = list(itertools.chain(*entities_ordered))
        
        # Create streets dict to store streets with multiple options
        streets = dict()
        # Flag
        MATCHED_FUZZY = False
        
        filtered_city_subset = filtered_subset
        
        # Iterate over entities
        for entity in entities_ordered:
            ent_type, name = entity

            res_row = self.df_string_match(name, filtered_subset)
            
            # If no result, try on full set of type subset
            if len(res_row['entity'].unique()) == 0:
                if ent_type in self.kb_types_subsets.keys():
                    res_row = self.df_string_match(name, self.kb_types_subsets[ent_type])
                    
                # If no subset or nothing has been found, try on full kb
                if len(res_row['entity'].unique()) == 0:
                    res_row = self.df_string_match(name, self.kb)
            
            
            # If no items in res_row, try fuzzy matching in filtered_subset
            if len(res_row['entity'].unique()) == 0:
                
                # Trye fuzzy search on filtered_subset
                res_candidate_fuzz = self.fuzzy_search_fields(name, filtered_subset, scorer=scorer)
                
                if type(res_candidate_fuzz) == bool:
                    if ent_type in self.kb_types_subsets.keys():
                        res_candidate_fuzz = self.fuzzy_search_fields(name, self.kb_types_subsets[ent_type], scorer=scorer)
                        
                    # If not in kb_types_subsets, try on full kb
                    if type(res_candidate_fuzz) == bool:
                        res_candidate_fuzz = self.fuzzy_search_fields(name, self.kb, scorer=scorer)
                        
                # If not False it must be df with matches
                if type(res_candidate_fuzz) != bool:
                    res_row = res_candidate_fuzz
                    MATCHED_FUZZY = True
            
            
            # If more than one candidate check if (alt) labels are in text
            # (in case the NER did not capture complete reference)
            if len(res_row['entity'].unique()) > 1:
                res_row_filtered = [(idx, cand) for idx, cand in res_row[['entityLabel']].dropna().iterrows() if cand[0] in text]
                res_row_filtered += [(idx, cand) for idx, cand in res_row[['entityAltLabel']].dropna().iterrows() if cand[0] in text]
                if len(res_row_filtered) != 0:
                    res_row = res_row.loc[[i[0] for i in res_row_filtered]]
                    
            
            # If still more than one candidate; check if location can be determined by street
            if len(res_row['entity'].unique()) > 1:
                # If it's a street, add it to street list.
                if ent_type == 'STREET':
                    streets[name] = res_row[['entity', 'entityLabel']].values.tolist()
                    
                # Only works if there is a street already found and current ent_type is not STREET
                if ent_type != 'STREET' and len(streets.keys()) > 0:
                    # Iterate over street_name (i.e. original street names in text) from streets (dict)
                    for street_name, street_candidates in streets.items():
                        # Iterate over street options 
                        for street_id, wd_street_name in street_candidates:
                            
                            # Check if street is in locstreetlabel
                            
                            # Check location and street column for street_id
                            res_row_filtered = res_row[res_row['entityStreet'] == street_id | res_row['entityLocation'] == street_id]
                            # If no result, try streetname on address
                            if len(res_row_filtered) == 0:
                                res_row_filtered[res_row['entityAddress'].str.contains(wd_street_name, na=False)]
                            # If no result, try on description
                            if len(res_row_filtered) == 0:
                                res_row_filtered[res_row['entityDescription'].str.contains(wd_street_name, na=False)]
                            
                            
                            # If result, add street to linked_entities dict, and update streets dict.
                            if len(res_row_filtered['entity'].unique()) >= 1:
                                self.linked_entities['STREET'][street_name] = {'name' : wd_street_name, "wikidata_id" : street_id}
                                streets[street_name] = [[street_id, wd_street_name]]
                                # Set ent_candidate_filtered as ent_candidate
                                res_row = res_row_filtered
                                
                                break
                            
                        # HIER NOG FF NAAR KIJKEN, OF DIT ZO GOED GAAT...
                        if len(res_row['entity'].unique()) == 1:
                            break
            
            
            # FUZZY MATCHING FILTER
            # If more than one candidate, do fuzzy match on res_row items
            # (if candidates where not found already by fuzzy matching)
            if len(res_row['entity'].unique()) > 1:
                if MATCHED_FUZZY == False:
                    # First fuzzy match on label and altlabel
                    fuzzy_search_list = res_row['entityLabel'].tolist() + res_row['entityAltLabel'].unique().tolist()
                    best_candidate = self.do_fuzzy_search(name, fuzzy_search_list, scorer=scorer)
                    # If no best match, try on description
                    if best_candidate == False:
                        best_candidate = self.do_fuzzy_search(name, res_row['entityDescription'].dropna().unique().tolist(), scorer=scorer)
                
                    # Get best option if there is a best_candidate
                    if best_candidate != False:
                        res_row = res_row[(res_row['entityLabel'] == best_candidate) | (res_row['entityAltLabel'] == best_candidate) | (res_row['entityDescription'] == best_candidate)]
            
            
            
            
            # If one option remains, link to that option. Add to linked_entities dict.
            if len(res_row['entity'].unique()) == 1:
                # Make key for ent_type, if not exists.
                if ent_type not in self.linked_entities.keys():
                    self.linked_entities[ent_type] = dict()
                # Add linked entity to dict
                self.linked_entities[ent_type][name] = {'name' : res_row['entityLabel'].values[0],
                                                        'wikidata_id' : res_row['entity'].values[0]}
                
                # If ent_type is street, add to streets dict as well (for internal use in function)
                if ent_type == 'STREET':
                    streets[name] = res_row[['entity', 'entityLabel']].values.tolist()
            
            # If still more than one option, add options to link_entities_manual dict
            if len(res_row['entity'].unique()) > 1:
                self.link_entities_manual[name] = {'ent_type' : ent_type, 'options' : res_row['entity'].unique().tolist()}
            
            
            # If in the end no result, add item to wd_lookup
            if len(res_row) == 0:
                if ent_type not in self.wd_lookup.keys():
                    self.wd_lookup[ent_type] = dict()
                self.wd_lookup[ent_type][name] = {'name' : False, 'wikidata_id' : False}
            
        
        # Handle streets (dict) with multiple options at last
        for street_name, street_options in streets.items():
            # Streets with one option are allready in linked_entities dict
            if len(set([i[0] for i in street_options])) > 1:
                self.link_entities_manual[street_name] = {'ent_type' : ent_type, 'options' : [option[0] for option in street_options]}
        
        return self.linked_entities, self.link_entities_manual
    
    
    
    def get_type_subsets(self):
        'Create kb subset with mapping from self.kb_types_map'
        kb_types_subsets = dict()
        for ent_type, entityTypes in self.kb_types_map.items():
            kb_types_subsets[ent_type] = self.kb[self.kb.entityType.isin(entityTypes)]
        
        return kb_types_subsets
        
        
        
    def do_fuzzy_search(self, look_for, candidate_list, scorer=fuzz.ratio, n_options=1):
        'Do fuzzy search to "look_for" in "candidate_list" and return best match'
        candidate_list = [i for i in candidate_list if type(i) == str]
        # Do fuzzy search
        fuzzy_options = process.extract(look_for, candidate_list, scorer=scorer, limit=n_options)
        # Filter best options
        best_options = [option for option in fuzzy_options if option[1]  >= self.threshold]
        # If no results, return False
        if len(best_options) == 0:
            return False
        # If n_options is 1, return option as just one string
        if n_options == 1:
            return best_options[0][0]
        
        # Else return list with best options
        return best_options
            
    
    def fuzzy_search_fields(self, name, df, scorer):
        
        """
        Do fuzzy search first on labels, than on alt label than on 
        description. Return filterd df if result, else return False.
        
        """
        
        # Fuzzy match on subtypes first labels than alt labels
        res_candidate_fuzz = self.do_fuzzy_search(name, df['entityLabel'].unique().tolist(), scorer=scorer)
        if res_candidate_fuzz != False:
            return df[df['entityLabel'] == res_candidate_fuzz]
        
        # Try on altlabel if no results
        if res_candidate_fuzz == False:
            res_candidate_fuzz = self.do_fuzzy_search(name, df['entityAltLabel'].unique().tolist(), scorer=scorer)
            if res_candidate_fuzz != False:
                return df[df['entityAltLabel'] == res_candidate_fuzz]
        # Try on description if no result
        if res_candidate_fuzz == False:
            res_candidate_fuzz = self.do_fuzzy_search(name, df['entityDescription'].unique().tolist(), scorer=scorer)
            if res_candidate_fuzz != False:
                return df[df['entityDescription'] == res_candidate_fuzz]
        
        return res_candidate_fuzz
    
    
    
    def df_string_match(self, name, df, search_description=False):
        'String match on dataframe. No fuzzy matching'
        
        # EXACT MATCH - Try exact match first
        res_row = df[df['entityLabel'] == name]
        
        # Try contains for label
        if len(res_row) == 0:
            # CONTAINS - Try contains on label
            res_row = df[df['entityLabel'].str.contains(name, na=False)]
        
        if len(res_row) == 0:
            res_row = df[df['entityAltLabel'] == name]
        if len(res_row) == 0:
            res_row = df[df['entityAltLabel'].str.contains(name, na=False)]
            
        if search_description == True:
            if len(res_row) == 0:
                res_row = df[df['entityDescription'].str.contains(name, na=False)]
            
        return res_row
        
    
    def get_city_and_subset(self, city_locations, text, scorer=fuzz.ratio):
        'Find city or cities in kb, and create subset'
        if type(city_locations) == str:
            city_locations = city_locations
            
        cities_kb = self.kb_types_subsets['CITY']
        
        self.city_matches = dict()
        # Iterate over all locations, and link them to Wikidata
        for name in city_locations:
            
            # Try DataFrame match first (on cities_kb)
            res_row = self.df_string_match(name, cities_kb)
            # Than try full kb
            if len(res_row.entity.unique()) == 0:
                res_row = self.df_string_match(name, self.kb)
            
            
            # If more than one, see if label is mentioned in text (possibly not complete match by NER model)
            if len(res_row['entity'].unique()) > 1:
                # Collect all candidate entities which labels are metioned in text
                res_row_filtered = [(idx, cand) for idx, cand in res_row[['entityLabel']].dropna().iterrows() if cand[0] in text]
                # If no labels mentioned in text, try alt labels
                if len(res_row_filtered) == 0:
                    res_row_filtered = [(idx, cand) for idx, cand in res_row[['entityAltLabel']].dropna().iterrows() if cand[0] in text]
                
                # If labels are mentioned in text, narrow down res_row to these entities
                if len(res_row_filtered) != 0:
                    res_row = res_row.loc[[i[0] for i in res_row_filtered]]
            
            
            # FUZZY MATCHING
            # If still more than one option or zero options ...
            
            if len(res_row) == 0 or len(res_row['entity'].unique()) > 1:
                # First set lookup list to res_row options
                fuzzy_lookup_list = res_row['entityLabel'].tolist()
                
                # If no res_row options, take full city list
                if len(res_row['entity'].unique()) == 1:
                    fuzzy_lookup_list = cities_kb.entityLabel.tolist()
                
                # Else do fuzzy match
                fuzzy_result = self.do_fuzzy_search(name, fuzzy_lookup_list, scorer=scorer)
                
                # If fuzzymatching gives result, set res_row to this result
                if fuzzy_result != False:
                    res_row = cities_kb[cities_kb['entityLabel'] == fuzzy_result]
                    

            # If more than one result
            if len(res_row['entity'].unique()) > 1:
                # If more than one option, but same labels (in case of municipality and place)
                if len(res_row['entityLabel'].unique()) == 1:
                    # Prefer municipality
                    mun = res_row[res_row['entityType'] == 'Q2039348']
                    if len(mun['entity'].unique()) == 1:
                        res_row = mun
                    else:
                        if 'CITY' not in self.link_entities_manual:
                            self.link_entities_manual['CITY'] = dict()
                        
                        self.link_entities_manual['CITY'][name] = res_row['entity'].unique()
            
            
            # If one result, take this
            if len(res_row['entity'].unique()) == 1:
                self.city_matches[name] = {'name' : res_row['entityLabel'].values[0],
                                           'wikidata_id' : res_row['entity'].values[0]}
            
            
            # If still no result, add item to wd_lookup dict
            elif len(res_row['entity'].unique()) == 0:
                
                if 'CITY' not in self.wd_lookup.keys():
                    self.wd_lookup['CITY'] = {}
        
                self.wd_lookup['CITY'][name] = {'name' : False, 'wikidata_id' : False}
            
        
        # CREATE KB SUBSET BASED ON CITY
        # Index kb for city subset (if matches are found, else just set keep kb as subset)
        self.city_filter = [self.city_matches[name]['wikidata_id'] for name in city_locations if name in self.city_matches.keys()]
        
        if len(self.city_filter) != 0:
            # IF cities are found, create subset
            filtered_subset = self.kb[self.kb.entityCity.isin(self.city_filter) | self.kb.entityLocation.isin(self.city_filter) | self.kb.entityLocation.isin(self.city_filter) | self.kb.teritories.isin(self.city_filter)]
        
        else:
            # Else set filtered_subset as full kb
            filtered_subset = self.kb
            
        self.linked_entities['CITY'] = self.city_matches
        
        return filtered_subset