function [W,b,gamma,beta,costs,costsv,xaxis] = MiniBatchGD(X, Y,valx,valy, cycleparams, W, b, lambda,gamma,beta)

%% cyclic learning rate hyperparameters
N=size(X,2);
k=size(W,2);
costs=[];
costsv=[];
xaxis=[];
freq=10; %number of times the cost should be computed per cycle
nmin=cycleparams(1);
nmax=cycleparams(2);
ns=cycleparams(3); % bsize =100 --> 1 cycle=10 epochs
nbatch=cycleparams(4); %ns=k*10000/100 = k*100
epochs=cycleparams(5); %so that training stops after one cycle
lmax=cycleparams(6);
alpha=0.99;
%%
t=0;
for l=0:lmax
    for epoch=1:epochs
      for j=1:N/nbatch
        if (t >= 2*l*ns) && (t <= (2*l+1)*ns)
            nt=nmin + ((t-2*l*ns)*(nmax-nmin))/ns;
        end
        if ((2*l+1)*ns < t) && (2*(l+1)*ns >=t)
            nt=nmax - ((t-((2*l+1)*ns))*(nmax-nmin))/ns;
        end
        jstart = (j-1)*nbatch + 1;
        jend = j*nbatch;
        inds = jstart:jend;
        Xbatch = X(:, jstart:jend);
        Ybatch = Y(:, jstart:jend);
        [P,s,shat,mu,v]=EvaluateClassifier(Xbatch,W,b,gamma,beta); % K x n
        [gW,gb,ggamma,gbeta]=ComputeGradients(Xbatch,Ybatch,P,s,W,lambda,shat,mu,v,gamma,beta); 
        for i=1:k
            W{i}= W{i} - (nt * gW{i});
            b{i}= b{i} - (nt * gb{i});
        end
        for i=1:k-1
            gamma{i}= gamma{i} - (nt * ggamma{i});
            beta{i}= beta{i} - (nt * gbeta{i});
        end
        if t==0
            movingMu=mu;
            movingVar=v;
        end
        for i=1:size(mu,2)
          movingMu{i}=alpha*movingMu{i} + (1-alpha)*mu{i};
          movingVar{i}=alpha*movingVar{i} + (1-alpha)*v{i};
        end
        if mod(t,500)==0 % calculate cost 10 times per cycle, 1 cycle with ns=800 is 1600 updates
           costs=[costs, ComputeCost(Xbatch,Ybatch,W,b,lambda,gamma,beta,movingMu,movingVar)]; % which is 16 epochs when
           costsv=[costsv, ComputeCost(valx,valy,W,b,lambda,gamma,beta,movingMu,movingVar)];   % batchsize is 100
           xaxis=[xaxis, t];
        end
        t=t+1;
      end
     seed=randperm(size(X,2)); % shuffle order of data(g)
     X=X(:,seed);  
     Y=Y(:,seed);
    end
end
% costs=[costs, ComputeCost(X,Y,W,b,lambda)]; % which is 16 epochs when
% costsv=[costsv, ComputeCost(valx,valy,W,b,lambda)];   % when batchsize is 100
% xaxis=[xaxis, t];

end