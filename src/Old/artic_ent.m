function [ ae ] = artic_ent( F, num_s )
%Calculate articulation entropy given a feature matrix
%Input: F:     Acoustic feature matrix. Each column corresponds to a feature
%              vecotr
%       num_s: Number of samples in the bootsrap sampling. You have to
%              determine this value based on the minimal length of F in
%              your dataset. Usually 0.9*min_length.

for j = 1:50  %Bootstrap sampling
    ind = randperm(size(F,1));
    F1 = F(ind(1:num_s),:);
    entropy1(j) = MST_entropy(F1,0.99);
end
ae = mean(entropy1);

end

