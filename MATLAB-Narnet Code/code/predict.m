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
Atrx=Atr1;%Atrx��ԭʼ���ݵ����е�2000-5000
n=1;
num2=n;%
num=(ended-numtrain-start)/n;%num*num2Ԥ��ĵ���
ypred=[];
for tpred=1:num2
T = tonndata(Atrx,false,false);%
[x,xi,ai,t] = preparets(net,{},{},T);%׼��һ��������
[y,xf,af] = net(x,xi,ai);%����ѵ���õ�������Ԥ��

[netc,Xic,Aic] = closeloop(net,xf,af);%closeloop��ʲô
y2 = netc(cell(0,num),Xic,Aic);%cell��һ����������
y2pred = cell2mat(y2).';%cell2mat��ָ��cell����תΪmat����
Atrx=[Atrx;y2pred];
ypred = [ypred;y2pred];
end
%%
plot(xta((numtrain+start+1):(numtrain+start+num*num2)),ypred,'blue');%Ԥ���ߣ���ɫ(numtrain-6000)
hold on;
xlim([0 50]);%xlimָ�����귶Χ0-50
plot(xta(1:start),Atar(1:start),'color','black');%ԭʼ�ߣ���ɫ(1-start)
plot(xta(start:(numtrain+start)),Atar(start:(numtrain+start)),'color','red');%ԭʼ�ߣ���ɫ(start-numtrain+start)
fig=plot(xta((numtrain+start+1):1:ended),Atar((numtrain+start+1):1:ended),'color','black');%ԭʼ�ߣ����ߣ���numtrain+1-6000��
%imagefilename = sprintf ( '%s %i', 'fig', '2' );
%saveas(fig,imagefilename);
