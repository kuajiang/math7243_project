# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:42:57 2023

@author: 22615
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import ast
import re
from PIL import Image, ImageDraw

def getTable(path):
    hemorrhage_labels = pd.read_csv(path+'hemorrhage-labels.csv')
    hemorrhage_labels['Im'] = hemorrhage_labels['Image']+'.jpg'
    
    #Results_Epidural = pd.read_csv(path+'/Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv')
    #Results_Epidural['Im'] = Results_Epidural['Origin']
    
    #pidural = pd.merge(hemorrhage_labels,Results_Epidural,on='Im',how='inner')
    #return Epidural

    Results_Intraparenchymal = pd.read_csv(path+'Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268.csv')
    Results_Intraparenchymal['Im'] = Results_Intraparenchymal['Origin']
    Intraparenchymal = pd.merge(hemorrhage_labels,Results_Intraparenchymal,on='Im',how='inner')
    return Intraparenchymal
    
def showPic(table,picPath,n_pic=5):
    index = 1
    for _ in range(len(table)):
        pic_bone = mpl.image.imread(picPath+'/brain_bone_window/'+table['Im'][_])
        pic_brain = mpl.image.imread(picPath+'/brain_window/'+table['Im'][_])
        pic_max = mpl.image.imread(picPath+'/max_contrast_window/'+table['Im'][_])
        pic_subdural = mpl.image.imread(picPath+'/subdural_window/'+table['Im'][_])
               
        if _<n_pic:            
            xlabel = []
            ylabel = []
            if type(table['All Labels'][_])==type(np.nan):
                print(table['All Labels'][_],"No correct label")
            else:
                for each in re.findall( r'\{[^}]*\}',table['All Labels'][_]):
                    points = ast.literal_eval(each)
                    xlabel.append(pic_bone.shape[0]*points['x'])
                    ylabel.append(pic_bone.shape[0]*points['y'])
            
            plt.figure(figsize=(8,8))
            plt.subplot(n_pic,4,index)
            plt.scatter(xlabel,ylabel,color='red',marker='.',linewidths=0.5)    
            plt.imshow(pic_bone)
            index+=1
            plt.subplot(n_pic,4,index)
            plt.scatter(xlabel,ylabel,color='red',marker='.',linewidths=0.5)    
            plt.imshow(pic_brain)
            index+=1
            plt.subplot(n_pic,4,index)
            plt.scatter(xlabel,ylabel,color='red',marker='.',linewidths=0.5)    
            plt.imshow(pic_max)
            index+=1
            plt.subplot(n_pic,4,index)
            plt.scatter(xlabel,ylabel,color='red',marker='.',linewidths=0.5)    
            plt.imshow(pic_subdural)
            index+=1
            plt.show()
            
    return None


def showLabel(table,picPath,n_pic=5):
    index = 1
    for _ in range(len(table)):
        pic_brain = mpl.image.imread(picPath+'brain_window/'+table['Im'][_])
        if _<n_pic:
            xlabel = {'All Labels':[],'Majority Label':[],'Correct Label':[]}
            ylabel = {'All Labels':[],'Majority Label':[],'Correct Label':[]}
            for key in list(xlabel.keys()):
                if type(table[key][_])==type(np.nan):
                    print('No',key)
                else:
                    for each in re.findall( r'\{[^}]*\}',table['All Labels'][_]):
                        points = ast.literal_eval(each)
                        xlabel[key].append(pic_brain.shape[0]*points['x'])
                        ylabel[key].append(pic_brain.shape[0]*points['y'])
                        
            plt.figure(figsize=(8,8))
            plt.subplot(n_pic,3,index)
            plt.scatter(xlabel['All Labels'],ylabel['All Labels'],color='red',marker='.',linewidths=0.5)    
            plt.imshow(pic_brain)
            index+=1
            plt.subplot(n_pic,3,index)
            plt.scatter(xlabel['Majority Label'],ylabel['Majority Label'],color='blue',marker='.',linewidths=0.5)    
            plt.imshow(pic_brain)
            index+=1
            plt.subplot(n_pic,3,index)
            plt.scatter(xlabel['Correct Label'],ylabel['Correct Label'],color='orange',marker='.',linewidths=0.5)    
            plt.imshow(pic_brain)
            index+=1
                
    return None
    
def showLabelpic(table,n_pic=10):
    for _ in range(len(table)):
        #pic_brain = mpl.image.imread('./renders/intraparenchymal/intraparenchymal/brain_window/'+table['Im'][_])
        pic_brain = mpl.image.imread('./renders/intraparenchymal/intraparenchymal/brain_window/'+table['Origin'][_])
        if _<n_pic:
            string = table['All Labels'][_].replace("[]","")
            #string = table['Majority Label'][_].replace("[]","")
            list_pattern = r'\[[^\[\]]*\]'
            valid_lists = re.findall(list_pattern, string)
            circles = [x for x in map(ast.literal_eval, valid_lists) if isinstance(x, list)]
            
            length = [len(points) for points in circles]
            print(length)
            plt.figure(figsize=(8,8))
            plt.imshow(pic_brain)
            for points in circles:
                xlabel = [p['x']*512 for p in points]
                ylabel = [p['y']*512 for p in points]
                plt.scatter(xlabel, ylabel,marker='.',linewidths=0.5)
                #plt.scatter(xlabel, ylabel, color='red',marker='.',linewidths=0.5)            
    return None
'''
def table_label(table,path='./label'):
    for _ in range(len(table)):
        string = table['All Labels'][_].replace("[]","").replace('\'\'',"")
        list_pattern = r'\[[^\[\]]*\]'
        valid_lists = re.findall(list_pattern, string)
        circles = [x for x in map(ast.literal_eval, valid_lists) if isinstance(x, list)]
        pic = np.full((512,512), False,dtype=bool)
        for points in circles:
            #print("here is the error", len(circles),table['Im'][_])
            #if len(points)>9) or len(circles)==:
            pic+=label_arr(points)
        img_pil = Image.fromarray(pic.astype(np.uint8) * 255)
        img_pil.save('./label/'+table['Image'][_]+'_label.jpg')
    return None
'''

def table_label(table,path='./label'):
    for _ in range(len(table)):
        pic = np.full((512,512), False,dtype=bool)
        pics =  table[table['Origin']==table['Origin'][_]]
        correct = pics['Correct Label'].tolist()
        for i in range(len(pics)):
            if type(correct[i])==type(np.nan):
                string = pics['Majority Label'].tolist()[i]
                if type(string)==type(np.nan):
                    img_pil = Image.fromarray(pic.astype(np.uint8) * 255)
                    img_pil.save('./label/label_'+table['Origin'][_][:-4]+'.png')
                    return None
                string = string.replace("[]","").replace('\'\'',"")
            else:
                string = correct[i].replace("[]","").replace('\'\'',"")
            list_pattern = r'\[[^\[\]]*\]'
            valid_lists = re.findall(list_pattern, string)
            circles = [x for x in map(ast.literal_eval, valid_lists) if isinstance(x, list)]         
            for points in circles:
                pic+=label_arr(points)
        img_pil = Image.fromarray(pic.astype(np.uint8) * 255)
        img_pil.save('./label/label_'+table['Origin'][_][:-4]+'.png')
    return None
           
def label_arr(points):
    if len(points)==0:
        return np.full((512,512), False,dtype=bool)
    xlabel = [p['x']*512 for p in points]
    ylabel = [p['y']*512 for p in points]
    img = Image.new('1', (512,512),color=0)
    draw = ImageDraw.Draw(img)
    xy = list(zip(xlabel, ylabel))
    draw.polygon(xy,fill=1)
    arr = np.array(img)   
    return arr

def checkphoto(table,name):
    pic_brain = mpl.image.imread('./renders/subdural/subdural/brain_window/'+name[6:-4]+'.jpg')        
    plt.figure(figsize=(8,8))
    plt.imshow(pic_brain)
    pics = table[table['Origin']==name[6:]]
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
            
#table = getTable('./Hemorrhage Segmentation Project/')
#table.to_csv('test.csv')
#showPic(table, './renders/epidural/epidural')
#showLabel(table, './renders/intraparenchymal/intraparenchymal/')
#showLabelpic(table)

table1 = pd.read_csv('./Hemorrhage Segmentation Project/Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv')
table1['Correct Label'] = table1['Correct Label'].replace('[[], []]',np.nan)
table1['Majority Label'] = table1['Majority Label'].replace('[]',np.nan)
table2 = pd.read_csv('./Hemorrhage Segmentation Project/Results_Intraparenchymal Hemorrhage Detection_2020-11-16_21.39.31.268.csv')

#table3 = intraventricular
table4 = pd.read_csv('./Hemorrhage Segmentation Project/Results_Subarachnoid Hemorrhage Detection_2020-11-16_21.36.18.668.csv')
table4['Correct Label'] = table4['Correct Label'].replace('[[]]',np.nan)
table4['Majority Label'] = table4['Majority Label'].replace('[]',np.nan)

table5 = pd.read_csv('./Hemorrhage Segmentation Project/Results_Subdural Hemorrhage Detection_2020-11-16_21.35.48.040.csv')
table5['Correct Label'] = table5['Correct Label'].replace('[[]]',np.nan)
table5['Majority Label'] = table5['Majority Label'].replace('[]',np.nan)
#showLabelpic(table2,15)

#table_label(table1)
#table_label(table2)
#table_label(table4)
table_label(table5)

checkphoto(table5, 'label_ID_1332d87fa.jpg')


