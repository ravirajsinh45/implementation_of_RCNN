import pandas as pd
import numpy as np
import cv2
from iou_calculation import iou_calc

cv2.setUseOptimized(True);
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()


def iou_filter(image_path,true_bb,thresh=0.5):
    
    '''
    arguments:
    1. image_path: path of images 
    2. true_bb: true labeled dataframe of image columns = ['filename','width','height','class','xmin','ymin','xmax','ymax']
    3. thresh: threshhold value for intersection over union(iou), by default 0.5
    
    returns:
    filtered_selective_search: rigion which iou is higher than threshhold value of given image and class of that object
    negative_example         : return region which iou is less than thresh value and not conflict with other object region
    
    '''
    
    img_name = image_path.split('/')[-1]
    
    img_bb = true_bb[true_bb['filename']==img_name].reset_index(drop=True)
    
    img = cv2.imread(image_path)
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    
    ss_bb = rects[:2000]
    
    filtered_selective_search = []

    negative_examples = []
    
    maybe_neagative = []
    
    # loop to compute iou for all label of perticular image
    for label in range(len(img_bb)):
        

        #unpacking cordinates
        true_xmin, true_ymin, true_width, true_height  = img_bb.loc[label,'xmin'], img_bb.loc[label,'ymin'], img_bb.loc[label,'xmax']-img_bb.loc[label,'xmin'], img_bb.loc[label,'ymax']-img_bb.loc[label,'ymin']
        class_of_label = img_bb.loc[label,'class']
        
        #loop to compute iou for all selective search of perticular label
        for j,rect in enumerate(ss_bb):
            calculating_iou_for_selectivesearch = iou_calc([true_xmin, true_ymin, true_width, true_height],rect)
            
            if calculating_iou_for_selectivesearch > thresh:
                filtered_selective_search.append([list(rect),class_of_label])
            
            elif calculating_iou_for_selectivesearch <0.2:
                maybe_neagative.append(list(rect))
    
    #removing duplicate entries
    
    def Remove(duplicate): 
        final_list = [] 
        for num in duplicate: 
            if num not in final_list: 
                final_list.append(num) 
        return final_list 

    maybe_neagative = Remove(maybe_neagative)
    filtered_selective_search = Remove(filtered_selective_search)
   

    #this is will use for background class for CNN which has iou less than 0.2, In paper it's 0.3 but in that also written that it's depends on dataset. 

    only_labels_of_filtered_selective_search = [x[0] for x in filtered_selective_search]

    for lab in maybe_neagative:
        condition = []    
        for true_lab in only_labels_of_filtered_selective_search:
            
            iou_for_negative_ex = iou_calc(true_lab,lab)
            
            condition.append(True) if iou_for_negative_ex <= 0.2  else condition.append(False)

        if False not in condition:
            negative_examples.append(lab)
    
    negative_examples = Remove(negative_examples)
    random_background_images_index = np.random.randint(low=0, high=len(negative_examples), size=2*len(only_labels_of_filtered_selective_search)) 
    random_background_images = [negative_examples[x] for x in random_background_images_index]

    
    return filtered_selective_search , Remove(random_background_images)
