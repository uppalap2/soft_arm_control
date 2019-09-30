%% generate data for Soft arm control with camera sensing 

P2_data_9cm = zeros(2700,6);
P2_data_15cm = zeros(2700,6);
P2_data_20cm = zeros(2700,6);
% P2_data_xcm = [Pb Pr L Px Py Pz]
Pb = 11:1:40;
Pr = 0:1:44;
%% Generate 9 cm data
i9 = 1;

L = 9e-2;
for ii = 1:length(Pb)
    for jj = 1:length(Pr)
        P_O_end = forward_kin(Pb(ii),Pr(jj),0,L);
        P2_data_9cm(i9,:) = [Pb(ii) Pr(jj) L P_O_end(1,1:3)];
        i9 = i9+1;
    end
end

L = 9e-2;
for ii = 1:length(Pb)
    for jj = 1:length(Pr)
        P_O_end = forward_kin(Pb(ii),-Pr(jj),0,L);
        P2_data_9cm(i9,:) = [Pb(ii) -Pr(jj) L P_O_end(1,1:3)];
        i9 = i9+1;
    end
end
        
save('P2_data_9cm.mat','P2_data_9cm')

%% Generate 15 cm data

i15 = 1;

L = 15e-2;
for ii = 1:length(Pb)
    for jj = 1:length(Pr)
        P_O_end = forward_kin(Pb(ii),Pr(jj),0,L);
        P2_data_15cm(i15,:) = [Pb(ii) Pr(jj) L P_O_end(1,1:3)];
        i15 = i15+1;
    end
end

L = 15e-2;
for ii = 1:length(Pb)
    for jj = 1:length(Pr)
        P_O_end = forward_kin(Pb(ii),-Pr(jj),0,L);
        P2_data_15cm(i15,:) = [Pb(ii) -Pr(jj) L P_O_end(1,1:3)];
        i15 = i15+1;
    end
end
        
save('P2_data_15cm.mat','P2_data_15cm')

%% Generate 20 cm data
i20 = 1;

L = 20e-2;
for ii = 1:length(Pb)
    for jj = 1:length(Pr)
        P_O_end = forward_kin(Pb(ii),Pr(jj),0,L);
        P2_data_20cm(i20,:) = [Pb(ii) Pr(jj) L P_O_end(1,1:3)];
        i20 = i20+1;
    end
end

L = 20e-2;
for ii = 1:length(Pb)
    for jj = 1:length(Pr)
        P_O_end = forward_kin(Pb(ii),-Pr(jj),0,L);
        P2_data_20cm(i20,:) = [Pb(ii) -Pr(jj) L P_O_end(1,1:3)];
        i20 = i20+1;
    end
end
        
save('P2_data_20cm.mat','P2_data_20cm')