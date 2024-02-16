# Dynamic Superiority Capture (DSC) clustering algorithm 

try:
    import pandas as pd
except ImportError:
    ! pip install pandas
    import pandas as pd




def dscn(df, text_column, unique_id_column, threshold = 0.3, nuclearisation=False, activate_char_level=False, remove_alphanumerics = False, remove_stand_alone_numbers = False,
         remove_punctuation = False, remove_stop_words_lemmatise = False, view_clean_sample =False):

    from collections import Counter
  
    df = df.sample(frac=1, axis=0,random_state=76) #BP76
    df.reset_index(drop=True, inplace=True)

    # Initialize new columns in the DataFrame for future assignment
    df[unique_id_column] = df[unique_id_column].astype(str)
    df['original_name'] = df[text_column]

    df["fam_id"] = None
    df["fam_prop"] = None
    df["searcher_prop"] = None
    # df["common_text"] = None


    #######################################################################
    # Cleaning
    ######################################################################

    if view_clean_sample:
        print("Sample of unclean text data\n")
        print(df[text_column].head(10))

    df[text_column] = df[text_column].str.lower()


    # Pattern to match stand-alone numbers
    stand_alone_numbers_pattern = r'\b\d+\b'

    # Pattern to specifically target alphanumeric strings
    alphanumeric_pattern = r'(?i)\b\w*[0-9]+\w*\b'

    # Conditional cleaning
    if remove_alphanumerics:
        df[text_column] = df[text_column].str.replace(alphanumeric_pattern, ' ', regex=True)

    
    if remove_stand_alone_numbers:
        df[text_column] = df[text_column].str.replace(stand_alone_numbers_pattern, ' ', regex=True)

    if remove_punctuation:
        import string
        punctuation_pattern = f"[{string.punctuation}]"
        df[text_column] = df[text_column].str.replace(punctuation_pattern, ' ', regex=True)

    if remove_stop_words_lemmatise:
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')  # Attempt to load spaCy model
        except ImportError:
            # Handle missing spaCy installation
            !pip install -q spacy
            import spacy
            print("downloading model for stop word removal and lemmatisation")
            !python -m spacy download en_core_web_sm
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            # Handle missing spaCy model
            print("downloading model for stop word removal and lemmatisation")
            !python -m spacy download en_core_web_sm
            nlp = spacy.load('en_core_web_sm')


        # Function to remove stop words and lemmatize
        def clean_stop_lemma(text):
            # Process the text with spaCy
            doc = nlp(text)
            # Generate lemmatized tokens that are not stop words
            lemmatized = [token.lemma_ for token in doc if not token.is_stop]
            # Join the lemmatized tokens back into a single string
            return ' '.join(lemmatized)
        
        df[text_column] = df[text_column].apply(clean_stop_lemma)
        



    df[text_column] = df[text_column].str.strip()

    if view_clean_sample:
        print("\n\nSample of clean text data\n")
        print(df[text_column].head(10))

    
    import sys
    def user_decision():
    # Prompt the user for input
        user_input = input("Do you want to proceed? (y/n): ").lower().strip()
        
        # Check the user's input and act accordingly
        if user_input == 'y':
            print("Proceeding with the rest of the code...")
            # Place the code here that you want to execute if the user decides to proceed
        elif user_input == 'n':
            print("Stopping the process.")
            sys.exit(0)# Exits the function to stop further execution
        else:
            print("Invalid input. Please enter 'y' for yes or 'n' for no.")

            
    if view_clean_sample:    
        user_decision()

    
    # DUPLICATES HANDLING

    # Identify duplicates (including the first occurrence)
    # duplicates = df[df.duplicated(text_column, keep=False)]

    # # Create a new DataFrame with the duplicates
    # duplicates_df = duplicates.copy()

    # # Remove duplicates from the original DataFrame, keeping the first occurrence
    # df = df.drop_duplicates(text_column, keep=False)
    # df.reset_index(inplace=True,drop=True)

    # # For each unique name in duplicates_df, assign a new fam_id and set fam_prop to 1
    # unique_names = duplicates_df[text_column].unique()

    # for name in unique_names:
    #     # Get the unique_id_column value for the first occurrence of the name
    #     unique_id_column_value = duplicates_df[duplicates_df[text_column] == name][unique_id_column].iloc[0]
        
    #     # Generate the fam_id using "fam_" + unique_id_column_value
    #     fam_id_value = "fam_" + unique_id_column_value
        
    #     # Assign fam_id and fam_prop to all rows with this name
    #     duplicates_df.loc[duplicates_df[text_column] == name, 'fam_id'] = fam_id_value
    #     duplicates_df.loc[duplicates_df[text_column] == name, 'fam_prop'] = -1
    #     duplicates_df.loc[duplicates_df[text_column] == name, 'searcher_prop'] = -1
        # duplicates_df.loc[duplicates_df[text_column] == name, 'common_text'] = 'duplicate'
        
    
    # duplicates_df["clean_text_data"] = duplicates_df[text_column]

    if activate_char_level:

        def longest_common_subsequence(str1, str2):
            # Create a 2D array to store lengths of longest common subsequence
            # Initialize all values to 0
            m, n = len(str1), len(str2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            # Fill dp array from bottom to top, right to left
            for i in range(m - 1, -1, -1):
                for j in range(n - 1, -1, -1):
                    if str1[i] == str2[j]:
                        dp[i][j] = 1 + dp[i + 1][j + 1]
                    else:
                        dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
            
            # Reconstruct LCS from dp array
            lcs = ""
            i, j = 0, 0
            while i < m and j < n:
                if str1[i] == str2[j]:
                    lcs += str1[i]
                    i += 1
                    j += 1
                elif dp[i + 1][j] > dp[i][j + 1]:
                    i += 1
                else:
                    j += 1
            
            return lcs


        

        def find_common_phrase_and_prop_ordered(df, index0, index1, name_of_column_to_cleanse):
            from collections import Counter
            # Tokenize the strings into characters, excluding spaces
            tokens_x0 = [char for char in df.loc[index0, name_of_column_to_cleanse] if char != ' ']
            tokens_x1 = [char for char in df.loc[index1, name_of_column_to_cleanse] if char != ' ']
            common_characters_ordered = longest_common_subsequence(tokens_x0,tokens_x1)
            counter_x0 = Counter(tokens_x0)
            counter_x1 = Counter(tokens_x1)
            union_max_occurrence = counter_x0 | counter_x1
            union_max_words_ordered = list(union_max_occurrence.elements())  # Convert the intersection back into a list
            common_phrase = ' '.join(common_characters_ordered)
            prop_common_words = len(common_characters_ordered) / len(union_max_words_ordered)

            return common_phrase, prop_common_words



    else: 

        # df[text_column] = df[text_column].str.split()
        def find_common_phrase_and_prop_ordered(df, index0, index1, name_of_column_to_cleanse):
            # Tokenize the names
            tokens_x0 = df.loc[index0, name_of_column_to_cleanse].split()
            tokens_x1 = df.loc[index1, name_of_column_to_cleanse].split()

            # Use Counter to count occurrences of each word
            counter_x0 = Counter(tokens_x0)
            counter_x1 = Counter(tokens_x1)

            # Find the intersection, respecting the minimum count between them
            intersection_counts = counter_x0 & counter_x1  # This finds the intersection of the two counters

            # Perform the union operation to get the maximum occurrence of any element across both multisets
            union_max_occurrence = counter_x0 | counter_x1

            common_words_ordered = list(intersection_counts.elements())  # Convert the intersection back into a list
            common_phrase = ' '.join(common_words_ordered)
            prop_common_words = len(common_words_ordered) / len(union_max_occurrence)
            # Return the common phrase and the proportion of common words
            return common_phrase, prop_common_words


    # Create an empty DataFrame to store results
    results_df_template = pd.DataFrame(columns=['common_words', 'prop_match'])

    # Iterate over each index in df
    for searcher_index in range(len(df)):
        # Use a copy of the template DataFrame to avoid accumulating results from previous iterations
        results_df = results_df_template.copy()

        for i in range(len(df)):
            if i != searcher_index:
                common_words, prop_match = find_common_phrase_and_prop_ordered(df, searcher_index, i, text_column)
                if not common_words :
                    common_words = 'no common words'
                # Create a new DataFrame for the current result
                new_row = pd.DataFrame({'common_words': [common_words], 'prop_match': [prop_match],'df_index': [i]}) # because i is the index
                # Concat df
                results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Find the maximum score in the DataFrame
        max_prop = results_df['prop_match'].max()


        # Get rows where the score column is equal to the maximum score
        max_score_rows = results_df[results_df['prop_match'] == max_prop]

        max_score_indices = max_score_rows['df_index'].unique()

        # comm_phrase = max_score_rows['common_words']

        # max_score_df_rows = df.iloc[max_score_indices]

        # This should handle all same Nones *
        if set(max_score_rows['common_words']) == {'no common words'}:
            fam_iden = "fam_" + str(df[unique_id_column][searcher_index])
            df.loc[searcher_index, "fam_id"] = fam_iden  
            df.loc[searcher_index, "fam_prop"] = 0.0
            df.loc[searcher_index, "searcher_prop"] = 0.0
            continue # no need to run the rest of the code

        # Present prop
        df.loc[searcher_index, "searcher_prop"] = max_prop

        all_same = max_score_rows['common_words'].nunique() == 1
    
        if all_same == True:
            
            # Check if max_score_indices already have a family
            existing_families = df.loc[max_score_indices, "fam_id"].dropna().unique()
            
            # All same can be true but existing family size can be 0, as a fresh family has not yet been formed, so a family of None
            if existing_families.size > 0:
                # Filter the DataFrame for existing_families
                filtered_df = df[df["fam_id"] == existing_families[0]]

                # Select the first non-null value from 'fam_prop' if it exists.. for new family's fam_prop could be None
                first_fam_prop_value = filtered_df['fam_prop'].dropna().iloc[0] if not filtered_df['fam_prop'].dropna().empty else None

                # Searcher capture -S0- fam_id will capture
                    
                if first_fam_prop_value is None or max_prop >= first_fam_prop_value:
                    
                    if pd.notnull(df['fam_id'][searcher_index]): # If i have assigned searcher a family id                   
                        fam_iden = df['fam_id'][searcher_index]                       
                        mask = df['fam_id'].isin(existing_families)
                        # Update searched fam_id to fam_iden
                        df.loc[mask, 'fam_id'] = fam_iden
                        df.loc[searcher_index, "fam_id"] = fam_iden # Redundant but just in case  
                        df.loc[max_score_indices, "fam_id"] = fam_iden
                        df.loc[searcher_index, "fam_prop"] = max_prop # Redundant but just in case 
                        df.loc[max_score_indices, "fam_prop"] = max_prop

                    elif pd.isnull(df['fam_id'][searcher_index]): # If searcher fam id is null. else could do but i wonna be sure
                        fam_iden = "fam_" + df[unique_id_column][searcher_index]
                        mask = df['fam_id'].isin(existing_families)
                        # Update fam_id to fam_iden where the condition is True
                        df.loc[mask, 'fam_id'] = fam_iden
                        df.loc[searcher_index, "fam_id"] = fam_iden  
                        df.loc[max_score_indices, "fam_id"] = fam_iden
                        df.loc[searcher_index, "fam_prop"] = max_prop
                        df.loc[max_score_indices, "fam_prop"] = max_prop

                # If searched is stronger, it will aquire searcher                   
                elif 0 <= max_prop < first_fam_prop_value:
                    # Inheret if it has some worth untill someone better comes and captures
                    df.loc[searcher_index, "fam_id"] = existing_families[0]
                    df.loc[searcher_index, "fam_prop"] = first_fam_prop_value

            elif existing_families.size == 0:
                # Create new family for all matched and index
                fam_iden = "fam_" + df[unique_id_column][searcher_index]
                df.loc[searcher_index, "fam_id"] = fam_iden 
                df.loc[max_score_indices, "fam_id"] = fam_iden
                df.loc[searcher_index, "fam_prop"] = max_prop
                df.loc[max_score_indices, "fam_prop"] = max_prop
                
        # If all_same == False
        else: # for common words that are not the same
            # Possible improvements character match, but ties can happen 2. GPT/LLM decision ?

            fam_iden = "fam_" + df[unique_id_column][searcher_index] # create fam_id for searcher
            most_frequent_or_first_mode_name = max_score_rows['common_words'].mode()[0] # Find the common words that appears most frequently

            # Filter the max_score_rows DataFrame to only include rows with the most frequent name
            filtered_df = max_score_rows[max_score_rows['common_words'] == most_frequent_or_first_mode_name]

            max_score_indices = filtered_df['df_index'].tolist()

            # Check if max_score_indices already have a family.. actually they should since all_same is false..but just in case
            existing_families = df.loc[max_score_indices, "fam_id"].dropna().unique()
        
    
            if len(existing_families) > 0: # there should be. since all_same is false, but just in case
                
                filtered_df = df[df["fam_id"] == existing_families[0]] 
                first_fam_prop_value = filtered_df['fam_prop'].dropna().iloc[0] 
            elif len(existing_families) == 0: 
                first_fam_prop_value = None

            # as well as len(existing_families) == 0 but will hold if first_fam_prop_value is None
            if first_fam_prop_value is None or max_prop >= first_fam_prop_value:
                
                # # Searcher capture -S0- will capture 
                # Searcher must also capture any existing family_id in the family as well 

                # Improvement
                # For index 2, index1 and 2 were fam_unique_id_column4, it seems index 5 replaced index2 without one
                # Update any existing family in the searched family id to searcher as well 
                fam_iden = "fam_" + df[unique_id_column][searcher_index] 
                mask = df['fam_id'].isin(existing_families)
                # Update fam_id to fam_iden where the condition is True
                df.loc[mask, 'fam_id'] = fam_iden
                df.loc[searcher_index, "fam_id"] = fam_iden 
                df.loc[max_score_indices, "fam_id"] = fam_iden # redundant but just in case
                df.loc[searcher_index, "fam_prop"] = max_prop
                df.loc[max_score_indices, "fam_prop"] = max_prop
                
            elif 0 <= max_prop < first_fam_prop_value :
                # Inheret if it has some worth 
                df.loc[searcher_index, "fam_id"] = existing_families[0] # IDEA 1 Which family id will it inheret if i select multiple families
                df.loc[searcher_index, "fam_prop"] = first_fam_prop_value


    
    df['clean_text_data'] = df[text_column]
    
    if nuclearisation:
        threshold = False
        # Check if all members in a fam_id are equal to 1
        def all_members_equal_one(fam_id, df):
            return (df[df['fam_id'] == fam_id]['searcher_prop'] == 1).all()

        # Process DataFrame based on condition
        for fam_id, group in df.groupby('fam_id'):
            if all_members_equal_one(fam_id, df):
                # If all members of fam_id are equal to 1, proceed with this strategy
                for (fam_id, text), subgroup in group.groupby(['fam_id', 'clean_text_data']):
                    first_index = subgroup.index[0]  # First index of the group
                    fam_iden = "fam_" + df.loc[first_index, unique_id_column]
                    df.loc[subgroup.index, "fam_id"] = fam_iden
                    # Process your logic here
            else: # this will not work for activate_char perhaps use common characters #IDEA
                # If not all members are equal to 1, use this strategy
                for (fam_id, searcher_prop), subgroup in group.groupby(['fam_id', 'searcher_prop']):
                    first_index = subgroup.index[0]  # First index of the group
                    fam_iden = "fam_" + df.loc[first_index, unique_id_column]
                    df.loc[subgroup.index, "fam_id"] = fam_iden


     
    if threshold:

        # Conditions for rows where search_prop < threshold
        mask = df['searcher_prop'] < threshold

        # Update fam_id for these rows
        df.loc[mask, 'fam_id'] = 'thres_fam_' + df.loc[mask, unique_id_column].astype(str)

        # Set fam_prop and searcher_prop to 0.0 for these rows
        df.loc[mask, ['fam_prop', 'searcher_prop']] = 0.0
    

    # df = pd.concat([df, duplicates_df], ignore_index=True)
    df.reset_index(inplace=True,drop=True)

    df['fam_count'] = df.groupby('fam_id')['fam_id'].transform('count') # Count family id

    df.sort_values(by='fam_id', inplace=True) # sort by family id
    df.reset_index(inplace=True, drop=True)
    df.drop([text_column ], axis=1, inplace=True) # ,"searcher_prop","fam_prop"
    df.rename(columns={'original_name': text_column}, inplace=True)
    return df

    # For more leniency clean data accordingly
# No unique id column fix
# Optimum theshold setting
