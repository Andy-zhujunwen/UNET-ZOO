# UNET-ZOO
including unet,unet++,attention-unet,r2unet,cenet,segnet ,fcn.

# ENVIRONMENT
window10(Ubuntu is OK)+pycharm+python3.6+pytorch1.3.1  

## HOW TO RUN:
python main.py --action train   #train the code
python main.py --action train&test   #train and test
python main.py --action test   #test 

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
