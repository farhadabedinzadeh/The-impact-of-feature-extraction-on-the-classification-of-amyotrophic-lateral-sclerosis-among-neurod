     clc;clear;close   
     warning off
%% '================ Written by Farhad AbedinZadeh ================'
%                                                                 %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  
%% 
path='.\GaitData\*.mat' ;
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    x =struct2array(load(fn));
    
    %% Seperate left and right foot gait signal
    left_foot=x(1,:);
    right_foot=x(2,:);

    %% apply wavelet and feature extraction
    wname='sym4'; %waveletName
    [c1,l1] = wavedec(left_foot,6,wname);
    [Ea1,Ed1] = wenergy(c1,l1);
    
    [c2,l2] = wavedec(right_foot,6,wname);
    [Ea2,Ed2] = wenergy(c2,l2);
    
   %%%%%%%%%%% feature %%%%%%%%%%%%%%%%%%%
   
    features(i,:)=[Ea1 Ed1 Ea2 Ed2];
    

end

% Prepare data for classifiers
output=[ones(13,1);2*ones(51,1)];

trainingData=[features output];
  
%% Linear Discriminant Analysis (LDA) with Cross-Validation(kfold 5)
[validationAccuracyLDA,SensitivityLDA,~] = LinearDiscriminant(trainingData);
%% Naïve Bayesian with Cross-Validation(kfold 5)
[validationAccuracyNB,SensitivityNB,~] = NaiveBayesian(trainingData);
%% Support Vector Machine(SVM) with Cross-Validation(kfold 5)
kfold=5;
[Accuracy,Sensitivity,Specificity,Precision,F1] = mSVM_opt(features,output,kfold);  

     acc=mean(Accuracy);
     sen=mean(Sensitivity);
     spec=mean(Specificity);
%% Result Table
tableeR=table(Accuracy',Sensitivity',Specificity',Precision',F1');
tableeR.Properties.VariableNames{1} = 'Accuracy';
tableeR.Properties.VariableNames{2} = 'Sensitivity';
tableeR.Properties.VariableNames{3} = 'Specificity';
tableeR.Properties.VariableNames{4} = 'Precision';
tableeR.Properties.VariableNames{5} = 'F1';
tableeR.Accuracy(6) = mean(Accuracy);
tableeR.Sensitivity(6) = mean(Sensitivity);
tableeR.Specificity(6) = mean(Specificity);
tableeR.Precision(6) = mean(Precision);
tableeR.F1(6) = mean(F1);
tableeR.F1(7) = std(F1);
tableeR.Precision(7) = std(Precision);
tableeR.Specificity(7) = std(Specificity);
tableeR.Sensitivity(7) = std(Sensitivity);
tableeR.Accuracy(7) = std(Accuracy);
