function [P,s,shat,mu,v] = EvaluateClassifier(X,W,b,gamma,beta,varargin)
    k=size(W,2);
    s=cell(1,k);
    shat=cell(1,k);
    shift=cell(1,k);
    n=size(X,2);
    
    if numel(varargin)==2
        mu=varargin{1};
        v=varargin{2};
    else
        mu=cell(1,k);
        v=cell(1,k);
    end    
    
    for l=1:k-1
        s{l}= W{l}*X + b{l}*ones(1,n); 
        if numel(varargin)~=2
            mu{l}=mean(s{l},2);
            v{l}=((var(s{l},0,2)*(n-1)) / n);
        end
        shat{l}=BatchNormalize(s{l},mu{l},v{l});
        shift{l}=repmat(gamma{l},1,size(shat{l},2)) .* shat{l} + repmat(beta{l},1,size(shat{l},2));
        X=max(0,shift{l});
    end
    s{k}= W{k}*X + b{k}*ones(1,n); 
    P=softmax(s{k});
end
