%% 多源域适应

clc
clear

%% load data


SD1=load('AIS_4_NGFs.mat');
SD1_data=double(SD1.data);
SD1_labels=double(SD1.labels);

SD2=load('ORS_4_DeCAF6.mat');
SD2_data=double(SD2.features);
SD2_labels=double(SD2.labels)+1;

TD=load('FuSAR_4_ResNet18.mat');
TD_data=double(TD.features);
TD_labels=double(TD.labels)+1;
%% select data
num_s1_source_data = 600; 
select_s1_source_data = 20;
num_s2_source_data = 600;
select_s2_source_data = 20;
num_target_data = 50;
select_train_val_data = 3;
class_num=size(unique(TD_labels),1);
rand('seed',1);
for t1 = 1:10
    l_s1_source_train_data(t1,:) = randperm(num_s1_source_data,select_s1_source_data);%600选取300
end

rand('seed',1);
for t1 = 1:10
    l_s2_source_train_data(t1,:) = randperm(num_s2_source_data,select_s2_source_data);%1200选取300
end

rand('seed',1);
for t2 = 1:10
    l_target_data(t2,:) = randperm(num_target_data);
    l_target_train_val_data(t2,:) = l_target_data(t2,1:select_train_val_data);%排序
    l_target_test_data(t2,:) = l_target_data(t2,select_train_val_data+1:end);%从50个中选取3个，
end

for num = 1:10
    n_s1_source_train_data = [l_s1_source_train_data(num,:) l_s1_source_train_data(num,:)+num_s1_source_data l_s1_source_train_data(num,:)+2*num_s1_source_data l_s1_source_train_data(num,:)+3*num_s1_source_data];
    n_s2_source_train_data = [l_s2_source_train_data(num,:) l_s2_source_train_data(num,:)+num_s2_source_data l_s2_source_train_data(num,:)+2*num_s2_source_data l_s2_source_train_data(num,:)+3*num_s2_source_data];
    n_target_train_val_data = [l_target_train_val_data(num,:) l_target_train_val_data(num,:)+num_target_data l_target_train_val_data(num,:)+2*num_target_data l_target_train_val_data(num,:)+3*num_target_data];
    n_target_test_data = [l_target_test_data(num,:) l_target_test_data(num,:)+num_target_data l_target_test_data(num,:)+2*num_target_data l_target_test_data(num,:)+3*num_target_data];
    
    source_s1_train_data = SD1_data(n_s1_source_train_data,:);
    source_s1_train_labels = SD1_labels(n_s1_source_train_data,:);
    source_s1_train_data = source_s1_train_data';%
    source_s1_train_data = source_s1_train_data ./ repmat(sqrt(sum(source_s1_train_data.^2)), size(source_s1_train_data, 1), 1);%归一化
    
    source_s2_train_data = SD2_data(n_s2_source_train_data,:);
    source_s2_train_labels = SD2_labels(n_s2_source_train_data,:);
    source_s2_train_data = source_s2_train_data';%
    source_s2_train_data = source_s2_train_data ./ repmat(sqrt(sum(source_s2_train_data.^2)), size(source_s2_train_data, 1), 1);
    
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
    all_C_s1=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
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
                    
                  %% training
                    s1_source_features = source_s1_train_data;
                    s2_source_features = source_s2_train_data;
                    target_features = target_train_val_data;
                    test_features = target_test_data;
                    s1_source_labels = source_s1_train_labels;
                    s2_source_labels = source_s2_train_labels;
                    target_labels = target_train_val_labels;
                    
                    kparam.kernel_type = 'gaussian';
                    [K_s1_s, param_s1_s] = getKernel(s1_source_features, kparam);
                    [K_s2_s, param_s2_s] = getKernel(s2_source_features, kparam);
                    [K_t, param_t] = getKernel(target_features, kparam);
                    
                    [K_s1_s_root, resnorm_s1_s] = sqrtm(K_s1_s); K_s1_s_root = real(K_s1_s_root);
                    [K_s2_s_root, resnorm_s2_s] = sqrtm(K_s2_s); K_s2_s_root = real(K_s2_s_root);
                    [K_t_root, resnorm_t] = sqrtm(K_t); K_t_root = real(K_t_root);
                    
                    n_s_s1 = size(K_s1_s, 1);
                    n_s_s2 = size(K_s2_s, 1);
                    n_t = size(K_t, 1);
                    n  = n_s_s1+n_s_s2+n_t;
                    
                    K_root  = [K_s1_s_root zeros(n_s_s1,n_s_s2) zeros(n_s_s1,n_t); zeros(n_s_s2, n_s_s1) K_s2_s_root zeros(n_s_s2, n_t); zeros(n_t, n_s_s1) zeros(n_t, n_s_s2) K_t_root];
                    
                    K_t_root_inv = real(pinv(K_t_root));
                    L_t_inv = [zeros(n_s_s1+n_s_s2, n_t);eye(n_t)] * K_t_root_inv;
                    
                    K_test      = getKernel(test_features, target_features, param_t);
                    
                    % =========================================================================
                    for c = 1:class_num
                        fprintf(1, '-- Class %d \n', c);
                    s1_source_binary_labels       = 2*(s1_source_labels == c) - 1;
                    s2_source_binary_labels       = 2*(s2_source_labels == c) - 1;
                    target_binary_labels       = 2*(target_labels == c) - 1;
                    
                    % training
                    [model, H, obj] = train_mshfa_mkl(s1_source_binary_labels,s2_source_binary_labels, target_binary_labels, K_root, param);
                    
                    % testing
                    
                    rho         = model.rho*model.Label(1);%-b
                    y_alpha     = zeros(n, 1);
                    y_alpha(full(model.SVs)) = model.sv_coef;
                    y_alpha     = y_alpha*model.Label(1);%w
                    y_alpha_t   = y_alpha(n_s_s1+n_s_s2+1:end);%目标域的数据的w
                    
                    tmp = (K_test*L_t_inv'*H*K_root);
                    
                    dec_values(:,c) = tmp*y_alpha + K_test*y_alpha_t - rho;
                    end
                     % display results
                    [~, predict_labels] = max(dec_values, [], 2);
                    acc_mshfa = sum(predict_labels(1:end) == target_test_labels(1:end))/length(target_test_labels);
                    fprintf('The accuracy = %f\n', acc_mshfa);                   
                          
                    if acc_mshfa >= best_acc
                        best_C = param.svm.C;
                        best_gamma = kparam.gamma;
                        best_w1= param.C_s1;
                        best_w2= param.C_s2;
                        best_acc = acc_mshfa;
                        best_pre=predict_labels;
                    end
                end
            end
        end
        
    end
    acc_2SD_MSHFA_test(num,1) = best_C;
    acc_2SD_MSHFA_test(num,2) = best_gamma;
    acc_2SD_MSHFA_test(num,3) = best_w1;
    acc_2SD_MSHFA_test(num,4) = best_w2;
    acc_2SD_MSHFA_test(num,5) = best_acc;
    pre_2SD_MSHFA_test(:,num) = best_pre(:);
end

mean_acc_mshfa = mean(acc_2SD_MSHFA_test(:,end));
std_acc_mshfa = std(acc_2SD_MSHFA_test(:,end));
fprintf('%.2f± %.2f\n',100*mean_acc_mshfa,100*std_acc_mshfa);
save('acc_result','acc_2SD_MSHFA_test');
save('prediction','pre_2SD_MSHFA_test');



