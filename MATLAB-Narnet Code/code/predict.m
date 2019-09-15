clc
load ('network_test.mat');
load pabc06.txt%change parameter
start=3375;%change parameter
ended=6001;
xta=pabc06(1:ended,1);%change parameter
Atar=pabc06(1:ended,3);%change parameter
numtrain=2000;%change parameter
Atr1=pabc06(start:(start+numtrain),3);%change parameter
%%
Atrx=Atr1;%Atrx是原始数据第三列的2000-5000
n=1;
num2=n;%
num=(ended-numtrain-start)/n;%num*num2预测的点数
ypred=[];
for tpred=1:num2
T = tonndata(Atrx,false,false);%
[x,xi,ai,t] = preparets(net,{},{},T);%准备一个空数组
[y,xf,af] = net(x,xi,ai);%利用训练好的神经网络预测

[netc,Xic,Aic] = closeloop(net,xf,af);%closeloop是什么
y2 = netc(cell(0,num),Xic,Aic);%cell是一种数据类型
y2pred = cell2mat(y2).';%cell2mat是指把cell类型转为mat类型
Atrx=[Atrx;y2pred];
ypred = [ypred;y2pred];
end
%%
plot(xta((numtrain+start+1):(numtrain+start+num*num2)),ypred,'blue');%预测线，蓝色(numtrain-6000)
hold on;
xlim([0 50]);%xlim指横坐标范围0-50
plot(xta(1:start),Atar(1:start),'color','black');%原始线，绿色(1-start)
plot(xta(start:(numtrain+start)),Atar(start:(numtrain+start)),'color','red');%原始线，红色(start-numtrain+start)
fig=plot(xta((numtrain+start+1):1:ended),Atar((numtrain+start+1):1:ended),'color','black');%原始线，黑线，（numtrain+1-6000）
%imagefilename = sprintf ( '%s %i', 'fig', '2' );
%saveas(fig,imagefilename);
