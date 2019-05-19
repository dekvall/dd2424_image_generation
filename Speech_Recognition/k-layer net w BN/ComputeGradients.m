function [gradW, gradb,gradGamma,gradBeta] = ComputeGradients(X, Y, P, s, W,lambda,shat,mu,v,gamma,beta)
    k=size(W,2);
    n=size(X,2);
    gradb=cell(1,k); %size(b) is number of layers
    gradW=cell(1,k);
    Ic=ones(n,1); % n x 1 ones
    
    g= -(Y-P);  %Y is K x N and P is K x N
    gradW{k}=((1/n)*(g*s{k-1}'))+(2*lambda*W{k});
    gradb{k}=((1/n)*(g*Ic));
    g=W{k}'*g;
    Indsk=s{k-1}>0;
    g=g.*Indsk;
    
    
    for l=k-1:-1:1
        gradGamma{l}=((g .* shat{l})*Ic)/n;
        gradBeta{l}=(g*Ic)/n;
        g=g .* (gamma{l}*Ic');
        
        g=BatchNormBackPass(g,s{l},mu{l},v{l});
        if l == 1
            si_1=X;
        else
            si_1=s{l-1};
        end
        gradW{l}=((g*si_1')/n)+(2*lambda*W{l});
        gradb{l}=(g*Ic)/n;
        if l > 1
           g=(W{l}')*g;
           si_1=si_1 > 0;
           g=g .* si_1;
        end
    end    
end
