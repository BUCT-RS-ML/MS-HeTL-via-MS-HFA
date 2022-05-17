
clc
clear
tic
addpath('D:\E盘\YGA_MSHFA_2nd\DATA\AIS\')
addpath('D:\E盘\YGA_MSHFA_2nd\DATA\ORS600\')
addpath('D:\E盘\YGA_MSHFA_2nd\DATA\SAR\')
%% data loading
SD1=load('AIS_NGF11_data.mat');
 SD1_data=double(SD1.AIS_NGF11_data);
SD1_labels=double(SD1.AIS_NGF11_labels);

SD2=load('ORS_JKF_alexnet_Decaf6.mat');
SD2_data=double(SD2.features);
SD2_labels=double(SD2.labels)+1;


TD=load('JKF_SAR_ResNet18_fc1.mat');
TD_data=double(TD.features);
TD_labels=double(TD.labels)+1;
%% sample selecting
num_ais_source_data = 600;
select_ais_source_data = 20;
num_ors_source_data = 600;
select_ors_source_data = 20;
num_target_data = 50;
select_train_val_data = 3;

rand('seed',1);
for t1 = 1:10
    l_ais_source_train_data(t1,:) = randperm(num_ais_source_data,select_ais_source_data);%600选取300
end

rand('seed',1);
for t1 = 1:10
    l_ors_source_train_data(t1,:) = randperm(num_ors_source_data,select_ors_source_data);%1200选取300
end

rand('seed',1);
for t2 = 1:10
    l_target_data(t2,:) = randperm(num_target_data);
    l_target_train_val_data(t2,:) = l_target_data(t2,1:select_train_val_data);%排序
    l_target_test_data(t2,:) = l_target_data(t2,select_train_val_data+1:end);%从50个中选取3个，
end

for num = 1:10
    n_ais_source_train_data = [l_ais_source_train_data(num,:) l_ais_source_train_data(num,:)+num_ais_source_data l_ais_source_train_data(num,:)+2*num_ais_source_data];
    n_ors_source_train_data = [l_ors_source_train_data(num,:) l_ors_source_train_data(num,:)+num_ors_source_data l_ors_source_train_data(num,:)+2*num_ors_source_data];
    n_target_train_val_data = [l_target_train_val_data(num,:) l_target_train_val_data(num,:)+num_target_data l_target_train_val_data(num,:)+2*num_target_data];
    n_target_test_data = [l_target_test_data(num,:) l_target_test_data(num,:)+num_target_data l_target_test_data(num,:)+2*num_target_data];
    
    
    source_ais_train_data = SD1_data(n_ais_source_train_data,:);
    source_ais_train_labels = SD1_labels(n_ais_source_train_data,:);
    source_ais_train_data = source_ais_train_data';%取转置
    source_ais_train_data = source_ais_train_data ./ repmat(sqrt(sum(source_ais_train_data.^2)), size(source_ais_train_data, 1), 1);%归一化
    
    source_ors_train_data = SD2_data(n_ors_source_train_data,:);
    source_ors_train_labels = SD2_labels(n_ors_source_train_data,:);
    source_ors_train_data = source_ors_train_data';%取转置
    source_ors_train_data = source_ors_train_data ./ repmat(sqrt(sum(source_ors_train_data.^2)), size(source_ors_train_data, 1), 1);
    
    target_train_val_data = TD_data(n_target_train_val_data,:);
    target_train_val_labels = TD_labels(n_target_train_val_data,:);
    target_train_val_data = target_train_val_data';
    target_train_val_data = target_train_val_data ./ repmat(sqrt(sum(target_train_val_data.^2)), size(target_train_val_data, 1), 1);
    
    target_test_data = TD_data(n_target_test_data,:);
    target_test_labels = TD_labels(n_target_test_data,:);
    target_test_data = target_test_data';
    target_test_data = target_test_data./ repmat(sqrt(sum(target_test_data.^2)), size(target_test_data, 1), 1);
    
    best_acc = 0;
    
    
    all_C =  [0.01,0.1,1,10,100];
    all_gamma =  [0.01,0.1,1,10,100];
    all_C_s1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];%设定目标域权重为1，择其它域的值表示其相对权重
    all_C_s2=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
    param.lambda = 1000;
    param.C_t = 1;
    param.mkl_degree = 1;
    
    for ii = 1:length(all_C)
        for jj = 1:length(all_gamma)
            for w1=1:length(all_C_s1)
                for w2=1:length(all_C_s2)
                    param.svm.C = all_C(ii);
                    kparam.gamma = all_gamma(jj);
                    param.C_s1 =  all_C_s1(w1);
                    param.C_s2 = all_C_s2(w2);
                    %% one vs one
                
                    %% 1.cargo and container
                    ais_source_features = source_ais_train_data(:,1:2*select_ais_source_data);
                    ors_source_features = source_ors_train_data(:,1:2*select_ors_source_data);
                    target_features = target_train_val_data(:,1:2*select_train_val_data);
                    test_features = target_test_data;
                    ais_source_labels = source_ais_train_labels(1:2*select_ais_source_data,:);
                    ors_source_labels = source_ors_train_labels(1:2*select_ors_source_data,:);
                    target_labels = target_train_val_labels(1:2*select_train_val_data,:);
                    
                    kparam.kernel_type = 'gaussian';
                    [K_ais_s, param_ais_s] = getKernel(ais_source_features, kparam);
                    [K_ors_s, param_ors_s] = getKernel(ors_source_features, kparam);
                    [K_t, param_t] = getKernel(target_features, kparam);
                    
                    [K_ais_s_root, resnorm_ais_s] = sqrtm(K_ais_s); K_ais_s_root = real(K_ais_s_root);
                    [K_ors_s_root, resnorm_ors_s] = sqrtm(K_ors_s); K_ors_s_root = real(K_ors_s_root);
                    [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
                    
                    n_s_ais = size(K_ais_s, 1);
                    n_s_ors = size(K_ors_s, 1);
                    n_t = size(K_t, 1);
                    n  = n_s_ais+n_s_ors+n_t;
                    
                    K_root  = [K_ais_s_root zeros(n_s_ais,n_s_ors) zeros(n_s_ais,n_t); zeros(n_s_ors, n_s_ais) K_ors_s_root zeros(n_s_ors, n_t); zeros(n_t, n_s_ais) zeros(n_t, n_s_ors) K_t_root];
                    
                    K_t_root_inv = real(pinv(K_t_root));
                    L_t_inv = [zeros(n_s_ais+n_s_ors, n_t);eye(n_t)] * K_t_root_inv;
                    
                    K_test      = getKernel(test_features, target_features, param_t);
                    
                    % =========================================================================
                    c = 1;
                    ais_source_binary_labels       = 2*(ais_source_labels == c) - 1;
                    ors_source_binary_labels       = 2*(ors_source_labels == c) - 1;
                    target_binary_labels       = 2*(target_labels == c) - 1;
                    
                    % training
                    [model, H, obj] = train_mshfa_mkl(ais_source_binary_labels,ors_source_binary_labels, target_binary_labels, K_root, param);
                    
                    % testing
                    
                    rho         = model.rho*model.Label(1);%-b
                    y_alpha     = zeros(n, 1);
                    y_alpha(full(model.SVs)) = model.sv_coef;
                    y_alpha     = y_alpha*model.Label(1);%w
                    y_alpha_t   = y_alpha(n_s_ais+n_s_ors+1:end);%目标域的数据的w
                    
                    tmp = (K_test*L_t_inv'*H*K_root);
                    dec_values12 = [];
                    dec_values12(:,:) = tmp*y_alpha + K_test*y_alpha_t - rho;
                    predict_12 = sign(dec_values12);
                    predict_12(find(predict_12 == 1)) = 1;
                    predict_12(find(predict_12 == -1)) = 2;
                    
                    % =========================================================================
                    
                    %% 2.cargo and tanker
                    ais_source_features = [source_ais_train_data(:,1:select_ais_source_data),source_ais_train_data(:,2*select_ais_source_data+1:end)];
                    ors_source_features = [source_ors_train_data(:,1:select_ors_source_data),source_ors_train_data(:,2*select_ors_source_data+1:end)];
                    target_features = [target_train_val_data(:,1:select_train_val_data),target_train_val_data(:,2*select_train_val_data+1:end)];
                    test_features = target_test_data;
                    ais_source_labels = [source_ais_train_labels(1:select_ais_source_data,:);source_ais_train_labels(2*select_ais_source_data+1:end,:)];
                    ors_source_labels = [source_ors_train_labels(1:select_ors_source_data,:);source_ors_train_labels(2*select_ors_source_data+1:end,:)];
                    target_labels = [target_train_val_labels(1:select_train_val_data,:);target_train_val_labels(2*select_train_val_data+1:end,:)];
                    
                    kparam.kernel_type = 'gaussian';
                    [K_ais_s, param_ais_s] = getKernel(ais_source_features, kparam);
                    [K_ors_s, param_ors_s] = getKernel(ors_source_features, kparam);
                    [K_t, param_t] = getKernel(target_features, kparam);
                    
                    [K_ais_s_root, resnorm_ais_s] = sqrtm(K_ais_s); K_ais_s_root = real(K_ais_s_root);
                    [K_ors_s_root, resnorm_ors_s] = sqrtm(K_ors_s); K_ors_s_root = real(K_ors_s_root);
                    [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
                    
                    n_s_ais = size(K_ais_s, 1);
                    n_s_ors = size(K_ors_s, 1);
                    n_t = size(K_t, 1);
                    n  = n_s_ais+n_s_ors+n_t;
                    
                    
                    K_root  = [K_ais_s_root zeros(n_s_ais,n_s_ors) zeros(n_s_ais,n_t); zeros(n_s_ors, n_s_ais) K_ors_s_root zeros(n_s_ors, n_t); zeros(n_t, n_s_ais) zeros(n_t, n_s_ors) K_t_root];
                    
                    K_t_root_inv = real(pinv(K_t_root));
                    L_t_inv = [zeros(n_s_ais+n_s_ors, n_t);eye(n_t)] * K_t_root_inv;
                    
                    K_test      = getKernel(test_features, target_features, param_t);
                    
                    % =========================================================================
                    c = 1;
                    ais_source_binary_labels       = 2*(ais_source_labels == c) - 1;
                    ors_source_binary_labels       = 2*(ors_source_labels == c) - 1;
                    target_binary_labels       = 2*(target_labels == c) - 1;
                    
                    % training
                    [model, H, obj] = train_mshfa_mkl(ais_source_binary_labels,ors_source_binary_labels, target_binary_labels, K_root, param);
                    
                    % testing
                    
                    rho         = model.rho*model.Label(1);
                    y_alpha     = zeros(n, 1);
                    y_alpha(full(model.SVs)) = model.sv_coef;
                    y_alpha     = y_alpha*model.Label(1);
                    y_alpha_t   = y_alpha(n_s_ais+n_s_ors+1:end);
                    
                    tmp = (K_test*L_t_inv'*H*K_root);
                    dec_values13 = [];
                    dec_values13(:,:) = tmp*y_alpha + K_test*y_alpha_t - rho;
                    predict_13 = sign(dec_values13);
                    predict_13(find(predict_13 == 1)) = 1;
                    predict_13(find(predict_13 == -1)) = 3;
                    
                    
                    %% 3.container and tanker
                    ais_source_features = source_ais_train_data(:,select_ais_source_data+1:end);
                    ors_source_features = source_ors_train_data(:,select_ors_source_data+1:end);
                    target_features = target_train_val_data(:,select_train_val_data+1:end);
                    test_features = target_test_data;
                    ais_source_labels = source_ais_train_labels(select_ais_source_data+1:end,:);
                    ors_source_labels = source_ors_train_labels(select_ors_source_data+1:end,:);
                    target_labels = target_train_val_labels(select_train_val_data+1:end,:);
                    
                    kparam.kernel_type = 'gaussian';
                    [K_ais_s, param_ais_s] = getKernel(ais_source_features, kparam);
                    [K_ors_s, param_ors_s] = getKernel(ors_source_features, kparam);
                    [K_t, param_t] = getKernel(target_features, kparam);
                    
                    [K_ais_s_root, resnorm_ais_s] = sqrtm(K_ais_s); K_ais_s_root = real(K_ais_s_root);
                    [K_ors_s_root, resnorm_ors_s] = sqrtm(K_ors_s); K_ors_s_root = real(K_ors_s_root);
                    [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
                    
                    n_s_ais = size(K_ais_s, 1);
                    n_s_ors = size(K_ors_s, 1);
                    n_t = size(K_t, 1);
                    n  = n_s_ais+n_s_ors+n_t;
                    
                   
                    K_root  = [K_ais_s_root zeros(n_s_ais,n_s_ors) zeros(n_s_ais,n_t); zeros(n_s_ors, n_s_ais) K_ors_s_root zeros(n_s_ors, n_t); zeros(n_t, n_s_ais) zeros(n_t, n_s_ors) K_t_root];
                    
                    K_t_root_inv = real(pinv(K_t_root));
                    L_t_inv = [zeros(n_s_ais+n_s_ors, n_t);eye(n_t)] * K_t_root_inv;
                    
                    K_test      = getKernel(test_features, target_features, param_t);
                    
                    % =========================================================================
                    c = 2;
                    ais_source_binary_labels       = 2*(ais_source_labels == c) - 1;
                    ors_source_binary_labels       = 2*(ors_source_labels == c) - 1;
                    target_binary_labels       = 2*(target_labels == c) - 1;
                    
                    % training
                    [model, H, obj] = train_mshfa_mkl(ais_source_binary_labels,ors_source_binary_labels, target_binary_labels, K_root, param);
                    
                    % testing
                    
                    rho         = model.rho*model.Label(1);
                    y_alpha     = zeros(n, 1);
                    y_alpha(full(model.SVs)) = model.sv_coef;
                    y_alpha     = y_alpha*model.Label(1);
                    y_alpha_t   = y_alpha(n_s_ais+n_s_ors+1:end);
                    
                    tmp = (K_test*L_t_inv'*H*K_root);
                    dec_values23 = [];
                    dec_values23(:,:) = tmp*y_alpha + K_test*y_alpha_t - rho;
                    
                    %标签确定
                    predict_23 = sign(dec_values23);
                    predict_23(find(predict_23 == 1)) = 2;
                    predict_23(find(predict_23 == -1)) = 3;
                    
                    
                    mshfa_test_predict = [predict_12,predict_13,predict_23];
                    mshfa_test_output = mode(mshfa_test_predict,2);
                    acc_mshfa = sum(mshfa_test_output(1:end) == target_test_labels(1:end))/length(target_test_labels);
                    
                    if acc_mshfa >= best_acc
                        best_C = param.svm.C;
                        best_gamma = kparam.gamma;
                        %best_lambda = param.lambda;
                        best_w1= param.C_s1;
                        best_w2= param.C_s2;
                        best_acc = acc_mshfa;
                        best_pre=mshfa_test_output;
                    end
                end
            end
        end
        
    end
    acc_2SD_MSHFA_test_lam1000w(num,1) = best_C;
    acc_2SD_MSHFA_test_lam1000w(num,2) = best_gamma;
    acc_2SD_MSHFA_test_lam1000w(num,3) = best_w1;
    acc_2SD_MSHFA_test_lam1000w(num,4) = best_w2;
    acc_2SD_MSHFA_test_lam1000w(num,5) = best_acc;
    pre_2SD_MSHFA_test_lam1000w(:,num) = best_pre(:);
end

mean_acc_mshfa = mean(acc_2SD_MSHFA_test_lam1000w(:,end));
std_acc_mshfa = std(acc_2SD_MSHFA_test_lam1000w(:,end));
fprintf('%.2f± %.2f\n',100*mean_acc_mshfa,100*std_acc_mshfa);
save('MSHFA_AIS_ngfs_ORS_GIST_SAR_ResNet18','acc_2SD_MSHFA_test_lam1000w');
save('pre_MSHFA_AIS_ngfs_ORS_GIST_SAR_ResNet18','pre_2SD_MSHFA_test_lam1000w');
toc



