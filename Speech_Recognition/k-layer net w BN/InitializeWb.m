function [W,b,gamma,beta] = InitializeWb(m,d)
    k=size(m,2); %nr of layers
    W=cell(1,size(m,2));
    b=cell(1,size(m,2));
    gamma=cell(1,size(m,2)-1);
    beta=cell(1,size(m,2)-1);
    rng(400);
    sigma=0; %1e-3 1e-4
    sigma2=1e-1;%1/sqrt(d)
    for i=k:-1:2
        W{i}=normrnd(0,1/sqrt(m(i-1)),m(i),m(i-1)); % k x m
        b{i}=zeros(m(i),1);
    end
    for i=k-1:-1:1
        gamma{i}=normrnd(1,0.0005,m(i),1);%ones(m(i),1);
        beta{i}=zeros(m(i),1);
    end
    W{1}=normrnd(0,1/sqrt(d),m(1),d);%*(2/sqrt(m(1))); % m x d random initialization of weights and biases 
    b{1}=zeros(m(1),1);
end