close all
%clear all

do_plot = 1; % 0 == no plots; 1 == all plots except boxplots; 2 == all plots
do_save = 1; % 0 == no plots; 1 == all plots except boxplots; 2 == all plots

runs = 1;
rr = [1:6];
nuser = 17;
if ~exist('data', 'var')
    data = cell(runs,1);
    for n = 1:runs
        data{n} = csvread(sprintf('data/ft_%d_result.csv', rr(n)),1);
    end
end

for n = 1:runs
    if n == 6
        nuser = 2;
    end
    aeD = rad2deg(abs(data{n}(:,1:10) - data{n}(:,11:20)));
    ae = abs(data{n}(:,1:10) - data{n}(:,11:20));
    se = ae.*ae;
    mse = mean(se);
    
    umse = zeros(nuser,10);
    umaeD = zeros(nuser,10);
    for u = 1:nuser
        ud = data{n}(data{n}(:,21)==(110+u),1:20);
        uae = abs(ud(:,1:10) - ud(:,11:20));
        uaeD = rad2deg(abs(ud(:,1:10) - ud(:,11:20)));
        use = uae.*uae;
        umse(u,:) = mean(use);    
        umaeD(u,:) = mean(uaeD);
    end
    
    % error for each angles and over all users
    fprintf('Run %i\n\tmse:', rr(n))
    fprintf('%f ', mse)
    fprintf('\n\ttotal mse: %f\n\n', mean(mse))     
    fprintf('\tmae:')
    fprintf('%f ', mean(aeD))
    fprintf('\n\ttotal mae: %f\n\n', mean(mean(aeD)))     
    fprintf('\tumse:')
    fprintf('%f ', mean(umse,2))
    fprintf('\n\ttotal umse: %f\n\n', mean(mean(umse,2)))     
    fprintf('\tumae:')
    fprintf('%f ', mean(umaeD,2))
    fprintf('\n\ttotal umae: %f\n\n', mean(mean(umaeD,2)))

    if do_plot || do_save
        
        figure('name', sprintf('%i: all PIP middle joint angles', rr(n)), 'units','normalized','outerposition',[0 0 1 1]);
        d = rad2deg((data{n}(data{n}(:,21)==120,[2,12])));
        range = [2990:2990+2200, 2991+2200:2991+4400, 10744:10744+2200, 17729:17729+2200, 21865:21865+2200, 25101:25101+2200];
        t = (1:length(d(range,1)))/100;
        subplot(211);
        plot(t,d(range,1),t,d(range,2),'LineWidth',2); grid on
        xlim([1;t(end)]); ylim([-2;125])
        ylabel('angle in degree');
        legend(sprintf('reference angles'), sprintf('predicted angles'))
        ylimits = ylim();
        rectangle('Position', [t(2200) ylimits(1)-1 t(2200) diff(ylimits)+2], 'FaceColor',[0.6 0.7 0.7 0.2],'EdgeColor','[0.1 0.1 0.1]', 'LineStyle', '--','LineWidth',2)
        rectangle('Position', [t(2200)*3 ylimits(1)-1 t(2200) diff(ylimits)+2], 'FaceColor',[0.6 0.7 0.7 0.2],'EdgeColor','[0.1 0.1 0.1]', 'LineStyle', '--','LineWidth',2)
        rectangle('Position', [t(2200)*5 ylimits(1)-1 t(2210) diff(ylimits)+2], 'FaceColor',[0.6 0.7 0.7 0.2],'EdgeColor','[0.1 0.1 0.1]', 'LineStyle', '--','LineWidth',2)
        text(t(925),105,'(1)','FontSize', 30, 'FontWeight', 'bold');text(t(950+2200),105,'(2)','FontSize', 30, 'FontWeight', 'bold');
        text(t(950+2200*2),105,'(3)','FontSize', 30, 'FontWeight', 'bold');text(t(950+2200*3),105,'(4)','FontSize', 30, 'FontWeight', 'bold');
        text(t(950+2200*4),105,'(5)','FontSize', 30, 'FontWeight', 'bold');text(t(950+2200*5),105,'(6)','FontSize', 30, 'FontWeight', 'bold');
        set(gca,'FontSize',34)
        title('PIP and MCP joint alngles for the middle finger')
        
%         fig = gcf;
%         fig.Color = 'white';
%         fig.InvertHardcopy = 'off';
%         if do_save
%             print_pdf(sprintf('figures/%i/middle_angles', rr(n)))
%         end
        
%        figure('name', sprintf('%i: raw sensor values', rr(n)), 'units','normalized','outerposition',[0 0 1 1]);
        if ~exist('edata', 'var')
            edata = csvread('data\user_number_1_to_6',1);
        end
        d = edata((edata(:,22)==120), 2:5)+1;
        t = (1:length(d(range,1)))/100;
        subplot(212);
        plot(t,d(range,1), t,d(range,2),t,d(range,3), t,d(range,4),'LineWidth',2); grid on
        %plot(d(range,:)+1,'LineWidth',2); grid on
        xlim([1;t(end)]); ylim([-0.1;2.4])
        ylabel('scaled raw value'); xlabel('time in [s]')
        legend('C1', 'C2', 'C3', 'C4')
        ylimits = ylim();
        rectangle('Position', [t(2200) ylimits(1)-1 t(2200) diff(ylimits)+2], 'FaceColor',[0.6 0.7 0.7 0.2],'EdgeColor','[0.1 0.1 0.1]', 'LineStyle', '--','LineWidth',2)
        rectangle('Position', [t(2200)*3 ylimits(1)-1 t(2200) diff(ylimits)+2], 'FaceColor',[0.6 0.7 0.7 0.2],'EdgeColor','[0.1 0.1 0.1]', 'LineStyle', '--','LineWidth',2)
        rectangle('Position', [t(2200)*5 ylimits(1)-1 t(2210) diff(ylimits)+2], 'FaceColor',[0.6 0.7 0.7 0.2],'EdgeColor','[0.1 0.1 0.1]', 'LineStyle', '--','LineWidth',2)
        text(t(925),2.2 ,'(1)','FontSize', 30, 'FontWeight', 'bold');text(t(950+2200),2.2,'(2)','FontSize', 30, 'FontWeight', 'bold');
        text(t(950+2200*2),2.2,'(3)','FontSize', 30, 'FontWeight', 'bold');text(t(950+2200*3),2.2,'(4)','FontSize', 30, 'FontWeight', 'bold');
        text(t(950+2200*4),2.2,'(5)','FontSize', 30, 'FontWeight', 'bold');text(t(950+2200*5),2.2,'(6)','FontSize', 30, 'FontWeight', 'bold');
        set(gca,'FontSize',34)
        title('Scaled capacitive sensing raw values')
        
        fig = gcf;
        fig.Color = 'white';
        fig.InvertHardcopy = 'off';
        if do_save
            print_pdf(sprintf('figures/%i/middle_angles_sp', rr(n)))
        end

        figure('name', sprintf('%i: mae angles', rr(n)), 'units','normalized','outerposition',[0 0 1 1]);
        bar(circshift(reshape(mean(aeD), [5,2]), 1)); grid on
        legend('PIP joint','MCP joint'); ylabel('degree'); xlabel('finger joints')
        title(sprintf('mean absolute error (mae) for all joint angles'))
        set(gca,'xticklabel',{'thumb'; 'index'; 'middle'; 'ring'; 'little'})
        set(gca,'FontSize',54)
        fig = gcf;
        fig.Color = 'white';
        fig.InvertHardcopy = 'off';
        if do_save
            print_pdf(sprintf('figures/%i/mae', rr(n)))
        end
        
        figure('name', sprintf('%i: boxplot mae angles', rr(n)), 'units','normalized','outerposition',[0 0 1 1]);
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
        set(gca,'FontSize',54)
        fig = gcf;
        fig.Color = 'white';
        fig.InvertHardcopy = 'off';
        if do_save
            print_pdf(sprintf('figures/%i/mae_bp', rr(n)))
        end

%         figure;
%         bar(mse); grid on
%         t = sprintf('MSE');
%         title(t)
%         %saveas(gcf,sprintf('figures/%i/%s.pdf', n,t))
%         if do_save
%             print_pdf(sprintf('figures/%i/%s', rr(n),t))
%         end
    
        umaeD2 = zeros(size(umaeD));
        umaeD2(:,1) = umaeD(:,5);
        umaeD2(:,2) = umaeD(:,10);
        umaeD2(:,3) = umaeD(:,1);
        umaeD2(:,4) = umaeD(:,6);
        umaeD2(:,5) = umaeD(:,2);
        umaeD2(:,6) = umaeD(:,7);
        umaeD2(:,7) = umaeD(:,3);
        umaeD2(:,8) = umaeD(:,8);
        umaeD2(:,9) = umaeD(:,4);
        umaeD2(:,10) = umaeD(:,9);
%         umaeD2(:,2:5) = umaeD(:,1:4);
%         umaeD2(:,6) = umaeD(:,10);
%         umaeD2(:,7:10) = umaeD(:,6:9);
        figure('name', sprintf('%i: umae angles and user', rr(n)), 'units','normalized','outerposition',[0 0 1 1]);
        bar(umaeD2); grid on
        title(sprintf('mean absolute error (mae) of each joint angle for all the participants'))
        legend('PIP joint thumb','MCP joint thumb','PIP joint index','MCP joint index','PIP joint middle','MCP joint middle','PIP joint ring','MCP joint ring','PIP joint little','MCP joint little');
        ylabel('degree'); xlabel('participants')
        set(gca,'FontSize',54)
        fig = gcf;
        fig.Color = 'white';
        fig.InvertHardcopy = 'off';
        if do_save
            print_pdf(sprintf('figures/%i/umae_single',rr(n)))
        end
        figure('name', sprintf('%i: umae', rr(n)),'units','normalized','outerposition',[0 0 1 1]);
        bar(mean(umaeD, 2)); grid on
        ylabel('degree'); xlabel('participants')
        title(sprintf('mean absolute error (mae) for the participants'))
        set(gca,'FontSize',54)
        fig = gcf;
        fig.Color = 'white';
        fig.InvertHardcopy = 'off';
        if do_save
            print_pdf(sprintf('figures/%i/umae', rr(n)))
        end
    
    end
    
    % errors groups for angles in 5 degress step
    if do_plot == 2 || do_save == 2
        figure('name', sprintf('%i: boxplot', rr(n)), 'units','normalized','outerposition',[0 0 1 1]); p = [1:5;6:10];
    end
    
    %ag = [5,10,1,6,2,7,3,8,4,9];
    ag = [5,1,2,3,4,10,6,7,8,9];
    j = {'PIP joint thumb', 'PIP joint index', 'PIP joint middle', 'PIP joint ring', 'PIP joint little', ...
        'MCP joint thumb', 'MCP joint index','MCP joint middle', 'MCP joint ring', 'MCP joint little'};
    for a = 1:10
        dl = discretize(rad2deg(data{n}(:,ag(a))), 0:5:90);
        da = sortrows([dl*5, abs(rad2deg(ae(:,ag(a))))]);
        %figure('name', sprintf('box %d',a));
        if do_plot == 2 || do_save == 2
            subplot(2,5,a)
            boxplot(abs(da(:,2)), da(:,1)); grid on
            ylim([-5;92])
            t = sprintf('%s', char(j(a)));
            title(t)
            %set(gca,'FontSize',18)
            fig = gcf;
            fig.Color = 'white';
            fig.InvertHardcopy = 'off';
        end
    end
    if do_save == 2
        print_pdf(sprintf('figures/%i/all_joints', rr(n)))
    end
end
