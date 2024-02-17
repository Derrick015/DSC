# Optimal threshold with sample

thres_list = []
perf_list = []
for i in range(0, 21):
    thres = (i / 20)
    df2 = dscn(df=df_sample, unique_id_column='ID', text_column = 'Mandarin', 
            nuclearisation=False, threshold=thres, activate_char_level=True) # 0.2
    df2['match_status_dscn'] = df2.apply(lambda row: 'match' if row['fam_count'] == row['fam_count_man'] else 'not match', axis=1)
    perf = df2['match_status_dscn'].value_counts(normalize=True).get('match', 0) * 100
    perf_list.append(perf)
    thres_list.append(thres)

# Zipping the lists together and creating a DataFrame
df_p = pd.DataFrame(list(zip(thres_list, perf_list)), columns=['Threshold', 'Performance'])
# Find the index of the maximum performance
max_perf_index = df_p['Performance'].idxmax()
# Retrieve the threshold with the maximum performance
max_threshold = df_p.loc[max_perf_index, 'Threshold']
# Retrieve the maximum performance value
max_performance = df_p.loc[max_perf_index, 'Performance']
# Display the result
print(f"The threshold with the maximum performance is {max_threshold} with a performance of {max_performance}%.")
