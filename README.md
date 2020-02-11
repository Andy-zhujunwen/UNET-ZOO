# UNET-ZOO
including unet,unet++,attention-unet,r2unet,cenet,segnet ,fcn.

# ENVIRONMENT
window10(Ubuntu is OK)+pycharm+python3.6+pytorch1.3.1  

## HOW TO RUN:
python main.py --action train   #train the code
python main.py --action train&test   #train and test
python main.py --action test   #test 

example:
python main.py --action train&test --arch UNet --epoch 21 --batch_size 21 

## RESULTS
after train and test,3 folders will be created,they are "result","saved_model","saved_predict".

### result folder:
in result folder,there are the logs and the line chart of metrics.
