function entropy = MST_entropy(X,a)
%INPUTS: 
%   X   : Data Matrix
%   a   : alpha parameter for Renyi Entropy


[N,d]=size(X);
gamma=d-d*a;

 weights = squareform(pdist(X,'euclidean'));
    maxWeight = 10 * max(max(weights));
    
    listOfEdges = [];
    currentMST = 0;
    
    [mstLength, edges] = mexMST(weights);
    [n,m]=size(edges);
    
    for i=1:n
        l(i)=weights(edges(i,1),edges(i,2));
    end
    L=sum(abs(l).^gamma);
     entropy=(1/(1-a))*(log(L/(N^a)));
end

