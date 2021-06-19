
import numpy as np
import pandas as pd
import json 

import pickle as p
# f = open('C:/Users/MEHAL/Downloads/FYP/MMDetection/results.pkl','rb')
f = open('/home/students/acct1002_15/mmdetection/results_3x_2.pkl','rb')

b = p.load(f)
b = list(b) #Convert b to list type to be converted to numpy type
b = np.array(b)

class_names = ('person', 'bicylce', 'car', 'motorcycle', 'bus', 'truck')


dict = {}
score_thr=0.3
for i in range (0, len(b)):
    bbox_result = b[i]
    bboxes = np.vstack(bbox_result)
    labels = [
       np.full(bbox.shape[0], i, dtype=np.int32)
       for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
 #   print(labels)
    for k, (bbox, label) in enumerate(zip(bboxes, labels)):
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        #print(label_text)
    dict[str(i)] ={}
#     print(len(labels))
    person = 0
    bicycle=0
    car=0
    motorcycle=0
    bus=0
    truck=0
    for j in range(0, len(labels)):
        if labels[j]==0:
            person +=1
        elif labels[j]==1:
            bicycle += 1
        elif labels[j]==2:
            car += 1
        elif labels[j]==3:
            motorcycle+= 1
        elif labels[j]==4:
            bus +=1
        elif labels[j]==5:
            truck+=1
    dict[str(i)]['person'] = person
    dict[str(i)]['bicycle'] = bicycle
    dict[str(i)]['car'] = car
    dict[str(i)]['motorcycle'] = motorcycle
    dict[str(i)]['bus'] = bus
    dict[str(i)]['truck'] = truck
    dict[str(i)]['total'] = person+bicycle+car+motorcycle+bus+truck
        

      
with open("/home/students/acct1002_15/MMDetection_count_objects/count_fulldataset.json", "w") as outfile:  
    json.dump(dict, outfile) 




