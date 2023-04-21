import ast
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shutil
import os

def read_segmentation_data():
    
    table1 = pd.read_csv('./labels/Results_Intraparenchymal_Hemorrhage_Detection_2020-11-16_21.39.31.268.csv')
    table1['Correct Label'] = table1['Correct Label'].replace('[[], []]',np.nan)
    table1['Majority Label'] = table1['Majority Label'].replace('[]',np.nan)
    table1['hemorrhage_type'] = 'intraparenchymal'

    table2 = pd.read_csv('./labels/Results_Intraventricular_Hemorrhage_Tracing_2020-09-28_15.21.52.597.csv')
    table2['Correct Label'] = table2['All Annotations'].replace('[[], []]',np.nan)
    table2['Majority Label'] = table2['All Annotations'].replace('[]',np.nan)
    table2['hemorrhage_type'] = 'intraventricular'

    table3 = pd.read_csv('./labels/Results_Subarachnoid_Hemorrhage_Detection_2020-11-16_21.36.18.668.csv')
    table3['Correct Label'] = table3['Correct Label'].replace('[[], []]',np.nan)
    table3['Majority Label'] = table3['Majority Label'].replace('[]',np.nan)
    table3['hemorrhage_type'] = 'subarachnoid'

    table4 = pd.read_csv('./labels/Results_Subdural_Hemorrhage_Detection_2020-11-16_21.35.48.040.csv')
    table4['Correct Label'] = table4['Correct Label'].replace('[[]]',np.nan)
    table4['Majority Label'] = table4['Majority Label'].replace('[]',np.nan)
    table4['hemorrhage_type'] = 'subdural'

    table5 = pd.read_csv('./labels/Results_Epidural_Hemorrhage_Detection_2020-11-16_21.31.26.148.csv')
    table5['Correct Label'] = table5['Correct Label'].replace('[[], []]',np.nan)
    table5['Majority Label'] = table5['Majority Label'].replace('[]',np.nan)
    table5['hemorrhage_type'] = 'epidural'

    table6 = pd.read_csv('./labels/Results_Multiple_Hemorrhage_Detection_2020-11-16_21.36.24.018.csv')
    table6['Correct Label'] = table6['Correct Label'].replace('[[], []]',np.nan)
    table6['Majority Label'] = table6['Majority Label'].replace('[]',np.nan)
    table6['hemorrhage_type'] = 'multi'

    table = pd.concat([table1, table2, table3, table4, table5, table6],axis=0)
    return table

def checkphoto2(table,name):
    pic_brain = mpl.image.imread('./renders/subdural/brain_window/'+name[6:-4]+'.jpg')        
    plt.figure(figsize=(8,8))
    plt.imshow(pic_brain)
    pics = table[table['Origin']==name[6:]]
    print(pics)
    for each in pics['Majority Label']:
    #for each in pics['Correct Label']:
        try:
            string = each.replace("[]","").replace('\'\'',"")
        except:
            string = '[{}]'
        list_pattern = r'\[[^\[\]]*\]'
        valid_lists = re.findall(list_pattern, string)
        circles = [x for x in map(ast.literal_eval, valid_lists) if isinstance(x, list)]       
        #length = [len(points) for points in circles]
        
        for points in circles:
            xlabel = [p['x']*512 for p in points]
            ylabel = [p['y']*512 for p in points]
            plt.scatter(xlabel, ylabel,marker='.',linewidths=0.5)
    print(name)
    plt.show()


# copy labled images to another folder ./labelled
def copy_labeled_images(table):
    for _, row in table.iterrows():
        image_name = row['Origin']
        image_name = image_name[:-4]
        hemorrhage_type = row['hemorrhage_type']
        brain_window_file = './renders/%s/brain_window/%s.jpg' % (hemorrhage_type, image_name)
        if not os.path.exists('./labelled/%s' % hemorrhage_type):
            os.makedirs('./labelled/%s' % hemorrhage_type)
        shutil.copyfile(brain_window_file, './labelled/%s/%s.jpg' % (hemorrhage_type, image_name))


def checkphoto(table, image_file_name):
    records = table[table['Origin']==image_file_name]

    image_name = image_file_name[:-4]
    hemorrhage_type = records.iloc[0]['hemorrhage_type']
    print("record_info", image_name, hemorrhage_type)
    brain_window_file = './renders/%s/brain_window/%s.jpg' % (hemorrhage_type, image_name)
    print(brain_window_file)
    pic_brain = mpl.image.imread(brain_window_file)        
    plt.figure(figsize=(8,8))
    plt.imshow(pic_brain)
    for each in records['Majority Label']:
    #for each in pics['Correct Label']:
        try:
            string = each.replace("[]","").replace('\'\'',"")
        except:
            string = '[{}]'
        list_pattern = r'\[[^\[\]]*\]'
        valid_lists = re.findall(list_pattern, string)
        circles = [x for x in map(ast.literal_eval, valid_lists) if isinstance(x, list)]       
        #length = [len(points) for points in circles]
        
        for points in circles:
            xlabel = [p['x']*512 for p in points]
            ylabel = [p['y']*512 for p in points]
            plt.scatter(xlabel, ylabel,marker='.',linewidths=0.5)
    # save labeled image
    target_file = './seg-label/%s/%s_labelled.jpg' % (hemorrhage_type, image_name)
    # plt.savefig(target_file)
    print(target_file)
    # release memory
    # plt.close()

    plt.show()

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
    # table = read_classify_lable()
    # check_file_exists(table)
    # return

    # do classification
    # table = pd.read_csv('./labels/hemorrhage-labels-exist.csv')
    





    table = read_segmentation_data()
    # copy_labeled_images(table)
    
    # image_name = 'ID_1332d87fa.jpg'
    # checkphoto(table, image_name)

    # for _, row in table.iterrows():
        # checkphoto(table, row['Origin'])
    
    # random_rows = table.sample(n=10)
    # for _, row in random_rows.iterrows():
        # checkphoto(table, row['Origin'])
    # checkphoto(table, 'ID_00eb6f7cc.jpg')
    checkphoto(table, 'ID_3a4e45124.jpg')


if __name__ == '__main__':
    main()