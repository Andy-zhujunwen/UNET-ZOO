# UNET-ZOO
including unet,unet++,attention-unet,r2unet,cenet,segnet ,fcn.

# ENVIRONMENT
window10(Ubuntu is OK)+pycharm+python3.6+pytorch1.3.1  

## HOW TO RUN:
The only thing you should do is enter the dataset.py and correct the path of the datasets.
then run ~
example:
```
python main.py --action train&test --arch UNet --epoch 21 --batch_size 21 
```
## RESULTS
after train and test,3 folders will be created,they are "result","saved_model","saved_predict".

### saved_model folder:
After training,the saved model is in this folder.

### result folder:
in result folder,there are the logs and the line chart of metrics.such as:
![image](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/linechart.png)

### saved_predict folder:
in this folder,there are the ouput predict of the saved model,such as:
![image](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/eye.png)
![image](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/lung.png)
![image](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master/cell.png)

### the datasets:
the Cell dataset(dsb2018)
link：https://pan.baidu.com/s/1BaVrzYdrSP78CwYaRzZr1w 
keyword：5l54 

the Liver dataset:
link：https://pan.baidu.com/s/1FljGCVzu7HPYpwAKvSVN4Q 
keyword：5l88 

the Cell dataset(isbi)
link：https://pan.baidu.com/s/1FkfnhU-RnYFZti62-f8AVA 
keyword：14rz

the Lung dataset:
link：https://pan.baidu.com/s/1sLFRmtG2TOTEgUKniJf7AA 
keyword：qdwo 

the Corneal Nerve dataset:
link：https://pan.baidu.com/s/1T3-kS_FgYI6DeXv3n1I7bA 
keyword：ih02

the Eye Vessels(DRIVE dataset)
link：https://pan.baidu.com/s/1UkMLmdbM61N8ecgnKlAsPg 
keyword：f1ek

the Esophagus and Esophagus Cancer dataset from First Affiliated Hospital of Sun Yat-sen University
link：https://pan.baidu.com/s/10b5arIQjNpiggwdkgYNHXQ 
keyword：hivm
