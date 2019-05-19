%% Loading all sets
[trainX,trainY,trainy]= LoadBatch('../train20.mat');
[testX,testY,testy]= LoadBatch('../test20.mat');

seed=randperm(size(trainX,2)); % shuffle order of data(g)
trainX=trainX(:,seed);  
trainY=trainY(:,seed);
trainy=trainy(seed);
seed=randperm(size(testX,2)); % shuffle order of data(g)
testX=testX(:,seed);  
testY=testY(:,seed);
testy=testy(seed);
sTest=10000;
sVal=3000;

% testX1=testX(:,1:sTest); testY1=testY(:,1:sTest); testy1=testy(1:sTest); %take 5k for test set
% testX(:,1:sTest)=[];testY(:,1:sTest)=[];testy(1:sTest)=[]; %remove the 5 k test set images from training
% 
% trainX=[trainX testX1]; trainY=[trainY testY1];trainy=[trainy;testy1];

valX=trainX(:,1:sVal); valY=trainY(:,1:sVal); valy=trainy(1:sVal); %take 5k for val set
trainX(:,1:sVal)=[];trainY(:,1:sVal)=[];trainy(1:sVal)=[]; %remove the 5 k val set images from training

%% Hyperparameter settings of network
m=[100,100,100,100,70,50,50,30];%[30,100,100,100,70,50,20,30];%8.94 %hidden nodes in layer 1
d=880; %dimension of the samples = 220
lambda=0;
N=size(trainX,2);
%%[220,200,180,160,140,130,120,110,100,90,80,70,60,50,20,30]; %hidden nodes in layer 1

%5*110 b=100 lambda=0.0001 l=10 nmin=1e-3; nmax=1e-1;[30,100,100,100,70,50,20,30]
%% Check analytically computed gradients against numerically computed (WORKS)
% steps=1e-5;
% eps=1e-3;
% [W,b,gamma,beta] = InitializeWb(m,d);
% [P,s,shat,mu,v] = EvaluateClassifier(trainX(1:20,1:5),W,b,gamma,beta);
% [gradW,gradb,gradgamma,gradbeta] = ComputeGradients(trainX(1:20,1:5), trainY(:,1:5),P, s, W,lambda,shat,mu,v,gamma,beta);
% [gradbNum, gradWNum,gradGNum,gradBNum] = ComputeGradsNumSlow(trainX(1:20,1:5), trainY(:,1:5), W, b,lambda, steps,gamma,beta);
% 
% for i=1:size(m,2)
%     relDiffNumb(i)=sum(abs(gradb{i} - gradbNum{i})/max(eps, sum(abs(gradb{i}) + abs(gradbNum{i}))));
%     relDiffNumW(i)=sum(abs(gradW{i} - gradWNum{i})/max(eps, sum(abs(gradW{i}) + abs(gradWNum{i}))));
%     centrDiffNumb(i)=sum(sum(abs(gradb{i}-gradbNum{i})));
%     centrDiffNumW(i)=sum(sum(abs(gradW{i}-gradWNum{i})));
% end
% 
% for i=1:size(gamma,2)
%     relDiffNumG(i)=sum(abs(gradgamma{i} - gradGNum{i})/max(eps, sum(abs(gradgamma{i}) + abs(gradGNum{i}))));
%     relDiffNumB(i)=sum(abs(gradbeta{i} - gradBNum{i})/max(eps, sum(abs(gradbeta{i}) + abs(gradBNum{i}))));
%     centrDiffNumG(i)=sum(sum(abs(gradgamma{i}-gradGNum{i})));
%     centrDiffNumB(i)=sum(sum(abs(gradbeta{i}-gradBNum{i})));
% end

%% check Mini batch gd without batchnormalization (WORKS)
nmin=1e-6;
nmax=1e-2;
ns=4*110;%5*450; % ns=500 & bsize =100 --> 1 cycle=10 epochs
l=5;
lambda=0.0001;
nbatch=100; %ns=k*10000/100 = k*100
epochs=(2*ns)/(11000/nbatch); 
cycleparams=[nmin,nmax,ns,nbatch,epochs,l];
allAcc=[];


for lambda=0.0001 %0.002:0.0004:0.0.004
    [W,b,gamma,beta] = InitializeWb(m,d);
    [Wstar,bstar,gstar,bestar,costs,costsv,xaxis]=MiniBatchGD(trainX,trainY,valX,valY,cycleparams,W,b,lambda,gamma,beta);
    accFinal=ComputeAccuracy(testX,testy,Wstar,bstar,gstar,bestar); %accuracy after training 
    allAcc=[allAcc, accFinal];
end
figure
plot(xaxis,costs)
ylim([0 4])
hold on;
plot(xaxis,costsv);
legend('training cost','validation cost')
