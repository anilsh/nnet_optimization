function compute_roc(dec,labels)

th = min(dec):0.001: max(dec);

tp = [];
fp = [];
for i = 1:length(th)
    plabel = dec>th(i);
    
    tp = [tp; sum(plabel(labels==1))/sum(labels==1)];
    fp = [fp; sum(plabel(labels==0))/sum(labels==0)]; 
end

figure;
plot(fp,tp, '-*');
xlabel('False Positive Rate');
ylabel('True Positive Rate');