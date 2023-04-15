import os
import pandas as pd

# read lable data
# set hemorrhage_type
def read_classify_lable():
    table = pd.read_csv('./labels/hemorrhage-labels.csv')

    # get first 100 rows
    # table = table.head(100)


    selected_columns = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural']
    
    # stat hemorrhage type
    print(table[selected_columns].sum(axis=0))

    # set normal as 1 where all other columns are 0
    table['normal'] = 1
    for column in selected_columns:
        table.loc[table[column] == 1, 'normal'] = 0
    selected_columns.append('normal')

    # set hemorrhage_type as the column name where the value of 1, else set as 'none'
    table['hemorrhage_type'] = table[selected_columns].idxmax(axis=1)
    table['count'] = table[selected_columns].sum(axis=1)
    idx_multi = table['count'] > 1
    table.loc[idx_multi, 'hemorrhage_type'] = 'multi'
    table = table.drop(columns=['count'])


    # pd.set_option('display.max_rows', 100)
    # print(table)

    return table


# check if image file exists
# and save rows with file exists to new csv file
def check_file_exists(table):
    # read all file names to a set
    file_names = set()
    selected_columns = ['epidural','intraparenchymal','intraventricular','subarachnoid','subdural', 'multi', 'normal']

    for column in selected_columns:
        path = './renders/%s/brain_window' % column
        before_len = len(file_names)
        for file in os.listdir(path):
            file_names.add(file)
            # print(file)
            # break
        print("count of ", column, len(file_names)-before_len)
    
    print("check file exists")

    # check Image in file_names
    table["file_exists"] = ["yes" if x+".jpg" in file_names else "no" for x in table["Image"]]

    # filter by file_exists
    table = table[table['file_exists'] == 'yes']


    # stat by hemorrhage_type
    print(table['hemorrhage_type'].value_counts())
    # save to a new csv file
    table.to_csv('./labels/hemorrhage-labels-exist.csv', index=False)
    

def main():
    # prepare date for calssification
    table = read_classify_lable()
    check_file_exists(table)



if __name__ == '__main__':
    main()