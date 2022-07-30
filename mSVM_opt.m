
function [Accuracy,Sensitivity,Specificity,Precision,F1] = mSVM_opt(feat,label,kfold)  
% Data preparation
fold   = cvpartition(label,'KFold',kfold);
Afold  = zeros(kfold,1);
for i = 1:kfold
  trainIdx = fold.training(i); testIdx = fold.test(i);
  xtrain   = feat(trainIdx,:); ytrain  = label(trainIdx);
  xvalid   = feat(testIdx,:);  yvalid  = label(testIdx); 
  % Train model
  Temp     = templateSVM('KernelFunction','rbf','KernelScale','auto');
  My_Model = fitcecoc(xtrain,ytrain,'Learners',Temp);
  % Test 
  pred = predict(My_Model,xvalid); clear My_Model
  % Accuracy
  Afold(i) = sum(pred == yvalid) / length(yvalid);  
% Overall accuracy
Acc = mean(Afold);

confmat = confusionmat(yvalid,pred); % where response is the last column in the dataset representing a class
TP = confmat(2, 2);
TN = confmat(1, 1);
FP = confmat(1, 2);
FN = confmat(2, 1);
Accuracy(i) = ((TP + TN) / (TP + TN + FP + FN))*100;
Sensitivity(i) = (TP / (FN + TP))*100;
Specificity(i) = (TN / (TN + FP))*100;
Precision(i) = (confmat(2,2)/(confmat(2,2)+confmat(1,2)))*100; 
F1(i) = (2*confmat(2,2))/(2*confmat(2,2)+confmat(2,1)+confmat(1,2))*100;
end
% fprintf('\n Accuracy: %g %%',100 * Acc);

fprintf('.......... SVM Result .......... \n')

fprintf('Mean of Validation Accuracy is %.2f', mean(Accuracy))
fprintf(' and Mean of Sensitivity is %.2f \n', mean(Sensitivity))

end

