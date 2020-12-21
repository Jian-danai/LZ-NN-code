# LZ-NN-code



Paper: Applications of neural networks to dynamics simulation of Landau-Zener transitions

link: https://www.sciencedirect.com/science/article/abs/pii/S030101041930802X

Pls contact me if you have any questions.

HomePage: https://jian-danai.github.io/

Email: yangbj@zju.edu.cn



#MATLAB-Narnet:

Steps to run the code:
Take pabc06 as an example.

* Copy the file(network_test.mat) which is the trained neural network into the folder "code" to replace the original file(also named network_test.mat) in the folder.

* Double click the file(predict.m) in the folder "code" to open the file using MATLAB.

* Change the parameters in the file(predict.m). The paremeters can be found in the file "/Networks&Parameters in predict.m/pabc06/06.txt". There are 6 parameters need to be changed in the first module of the code(which have been commented with "change parameter"). 

* Run the code and you will see the figure including predicted lines.

* Train_combined.m file is used to train the neural network.


#Python3-LSTM:

* Requirement: Python3, tensorflow, pandas, matplotlib, numpy, scipy

* The LSTM was trained in iMac. But it will also work well in Linux and Windows.

# Reference

Please cite this in your publication if our work helps your research. 

```
@article{yang2020applications,
  title={Applications of neural networks to dynamics simulation of Landau-Zener transitions},
  author={Yang, Bianjiang and He, Baizhe and Wan, Jiajun and Kubal, Sharvaj and Zhao, Yang},
  journal={Chemical Physics},
  volume={528},
  pages={110509},
  year={2020},
  publisher={Elsevier}
}
```

