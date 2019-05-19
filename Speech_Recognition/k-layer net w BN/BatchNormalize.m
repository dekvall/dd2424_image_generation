function [shat] = BatchNormalize(s,mean,var)
    eps=1e-6;
    shat=(diag(var+eps)^(-0.5))*(s-repmat(mean,1,size(s,2)));
end