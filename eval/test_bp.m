close all;

A=rand(size(aeD,1),5);
B=rand(size(aeD,1),5);

Ac{1}=aeD(:,5);
Ac{2}=aeD(:,1);
Ac{3}=aeD(:,2);
Ac{4}=aeD(:,3);
Ac{5}=aeD(:,4);
Bc{1}=aeD(:,10);
Bc{2}=aeD(:,6);
Bc{3}=aeD(:,7);
Bc{4}=aeD(:,8);
Bc{5}=aeD(:,9);

% prepare data
data2=vertcat(Ac,Bc);

xlab={'Thumb','Index','Middle','Ring','Little'};
col=[102,255,255, 200; 
    51,153,255, 200];
col=col/255;

multiple_boxplot(data2',xlab,{'PIP joint', 'MCP joint'},col')
ylabel('degree'); xlabel('finger joints')
title(sprintf('mean absolute error (mae) for all joint angles'))
grid on