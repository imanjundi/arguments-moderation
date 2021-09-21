import pandas as pd
from pathlib import Path

from moderator_intervention_full.args import DataTrainingArguments

THRESHOLD = 0.5

columns_mapping = {
    'Social Functions (Welcome, Ataboy, and thanks)? Yes/No': 'Social Functions',
    'Site Use Issues? Yes/No': 'Resolving Site Use Issues',
    'Organizing Discussion - direct to another post? Yes/No ': 'Organizing Discussion',
    'Policing (Civility, relevance, wrong venue)? Yes/No': 'Policing',
    'Out of bounds for agency? Yes/No': 'Keeping Discussion on Target',
    'Improving individual comment quality (user- specific)? Yes/No': 'Improving Comment Quality',
    'Broadening discussion (between users or bring other users in)? Yes/No': 'Broadening Discussion'
}
moderator_actions = list(columns_mapping.values())


def read_dataset_with_all_annotations(path: str):
    path = Path(path) / 'regulationroom'
    # read annotations
    df1 = pd.read_excel(path / 'original_data/08_03_2011/EOBR Comment Quality Coding All 8_3_11.xls')
    print(f"number of comments in EOBR {df1['comment ID'].nunique()}")
    df2 = pd.read_excel(path / 'original_data/10_04_2011/APR Moderator Comments Combined 10_4_11.xlsx')
    print(f"number of comments in APR {df2['comment_ID'].nunique()}")

    # read comments
    comments_df1 = pd.read_excel(path / 'Comment_Data_from_CeRI_4_3_2017.xlsx',
                                 sheet_name='Electronic On-Board Recorders')
    comments_df2 = pd.read_excel(path / 'Comment_Data_from_CeRI_4_3_2017.xlsx',
                                 sheet_name='Airline Passenger Rights')

    # map comment ids to text
    df1['comment parent content'] = df1['comment parent'].map(comments_df1.set_index('COMMENT ID')['COMMENT'])
    df2['comment_parent_content'] = df2['comment_parent'].map(comments_df2.set_index('COMMENT ID')['COMMENT'])

    if [x for x in df1['comment ID'].to_list() if x in df2['comment_ID'].to_list()]:
        print('duplicates in IDs!!! use in combination with type!')

    # join datasets
    df2.rename(columns=lambda x: x.replace('_', ' '), inplace=True)
    df1['type'] = 'EOBR'
    df2['type'] = 'APR'
    df = pd.concat([df1, df2], ignore_index=True, sort=False)
    print(f"number of comments in merged dataset {df.groupby(['comment ID', 'type']).ngroups}")

    # aggregate annotations
    df.rename(columns=columns_mapping, inplace=True)
    df[moderator_actions] = df[moderator_actions].replace({'Yes': 1, 'No': 0})
    return df


def read_dataset(path: str):
    df = read_dataset_with_all_annotations(path)
    df = df.groupby(['comment ID', 'type']).agg(
        {**{x: 'first' for x in df.columns.to_list() if x not in moderator_actions},
         **{x: 'mean' for x in moderator_actions}})
    print(f"number of comments after aggregation {df.shape[0]}")

    for x in moderator_actions:
        df[x + ' label'] = df[x].map(lambda x: 1 if x > THRESHOLD else 0)

    return df


def extract_quality_improved_ids(path: str):
    # return the comment IDs of airline passenger rights and electronic on board recorders
    df = read_dataset(path)
    apr_ids = []
    eobr_ids = []
    for _, row in df.iterrows():
        if row["Improving Comment Quality label"] == 1:
            if row["type"] == "APR":
                apr_ids.append(row["comment parent"])
            else:
                eobr_ids.append(row["comment parent"])
    return set(apr_ids), set(eobr_ids), set(list(df["comment parent"]))


def extract_improve_quality_df(data_args: DataTrainingArguments):
    # extract all IDs annotated with having been moderated with "improve quality"
    apr_ids, eobr_ids, annotated_ids = extract_quality_improved_ids(data_args.data_dir)
    df = read_dataset_complete(data_args)
    rules = ["Airline Passenger Rights", "Electronic On-Board Recorders"]
    if not data_args.in_domain:
        # remove all datapoints that are moderated but that are not annotated
        # (as we don't know the reason of moderation there)
        df = df[(df.type.isin(rules) &
                 df['MODERATED'] & df["COMMENT ID"].isin(annotated_ids)) | ~df['MODERATED']]
    else:
        # keep only in-domain (`rules` are the only ones that are annotated for quality moderation)
        df = df[(df.type.isin(rules) &
                 ((df['MODERATED'] & df["COMMENT ID"].isin(annotated_ids)) | ~df['MODERATED']))]
    quality_label = []
    for _, row in df.iterrows():
        if row.type == "Airline Passenger Rights" and row["COMMENT ID"] in apr_ids:
            quality_label.append(1)
        elif row.type == "Electronic On-Board Recorders" and row["COMMENT ID"] in eobr_ids:
            quality_label.append(1)
        else:
            quality_label.append(0)
    df["MODFORQUALITY"] = quality_label
    return df


def read_dataset_complete(data_args: DataTrainingArguments):
    # based on regulation_room_complete.ipynb
    xls = pd.ExcelFile(Path(data_args.data_dir) / 'regulationroom/Comment_Data_from_CeRI_4_3_2017.xlsx')
    dfs = [xls.parse(x).assign(type=x) for x in xls.sheet_names[1:]]
    for df in dfs:
        # one row has nan for user
        df.dropna(subset=['USER LOGIN'], inplace=True)
        df['MODERATOR'] = df['USER LOGIN'].map(lambda x: x.lower() == 'moderator')
        df['MODERATED'] = df['COMMENT ID'].map(lambda x: df[df['COMMENT PARENT'] == x]['MODERATOR'].any())

        # add comment parents to the dataset
        df['COMMENT PARENT 1'] = df['COMMENT PARENT']
        for i in range(1, data_args.comment_parents_num + 1):
            df[f'COMMENT PARENT {i} CONTENT'] = df[f'COMMENT PARENT {i}'].map(df.set_index('COMMENT ID')['COMMENT'])
            df[f'COMMENT PARENT {i} CONTENT'].fillna('', inplace=True)
            df[f'COMMENT PARENT {i} USER LOGIN'] = df[f'COMMENT PARENT {i}'].map(
                df.set_index('COMMENT ID')['USER LOGIN'])
            df[f'COMMENT PARENT {i + 1}'] = df[f'COMMENT PARENT {i}'].map(df.set_index('COMMENT ID')['COMMENT PARENT'])

    df = pd.concat(dfs, ignore_index=True, sort=False)
    # drop the many Unnamed columns with all nan values
    df.dropna(axis=1, how='all', inplace=True)
    return df
