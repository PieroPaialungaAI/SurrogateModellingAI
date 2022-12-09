clc
clear
R = [10]
k1=10000000
for r=R 
    q=BuildDefect_func(r);
    base_string = string('defect');
    num_defect = string(k1);
    end_point = string('.csv');
    tot_string = strcat(base_string,num_defect);
    tot_string = strcat(tot_string,end_point);
    csvwrite(tot_string,q);
    k1=k1+1
end
%%

function D = BuildDefect_func(R)
numr = 8;
E = 210e9;  % steel e.g. nu = 0.29, rho = 7800, E = 200-215e9; want approx 5920 m/s
nu = 0.29; 
rho = 7800;
fcen = 5e6;

smooth=1;     % for flat/smooth crack, we ignore surface roughness 
              % parameter setting
              % 0 for smooth
              % 1 for rough
            

CT= sqrt(0.5*E/(rho*(1+nu)));                   % Shear wave velocity CT
CL=sqrt(E*(1-nu)/(rho*(1+nu)*(1-2*nu)));        % Longitudinal velocity CL 
Lwavelength = CL/fcen;
Twavelength = CT/fcen;

% shear wavelength is between 0.6 and 0.9mm for typical frequencies. 
% 50 * 0.9 = 45mm
% We want the point separation to be 0.025mm. thus 50/0.025=2000 points
% should be enough. For compress ion wave incidence, correlation length limit
% is roughly lambda_p/2. Hence 50*(0.75 to 0.85mm) is also appropriate.

Length = 50e-3;           % the length of the statistically stable crack in m 
Finitelength = 6e-3;      % sectioned length for modelling
%Num_points = 2001;
Num_points = 2001;  % number of discrete points of height along 
                          % the crack.
% (NB Ogilvy useed 8000)
%Recall from the paper that one generates 2N+1 uncorrelated random 
%numbers v_i. The weights are generated using 2M+1 terms and the final set 
%of heights is given by 2L+1 where N = L + M                     

L = (Num_points-1)/2; 

dx = Length/(Num_points-1);  % the interval between adjacent points

% roughness parameters
RSh = R;                     % RMS / sigma: fraction of incident wavelength
Hco = 1/2;                   % Correlation length - no. of incident wavelengths

    RMS_H = Lwavelength/RSh;    % the RMS of height, m
    H_corr = Hco*Lwavelength;   % the height correlation length, m

    M =  floor(4*H_corr/(dx*sqrt(2)))+1;  

    % the value of M, page 262 in Ogilvy paper, and explained in my notebook.
    % The important fact is that 4 weight correlation lengths are sufficient.

    seed=1;   %used when generating random numbers, an integer. The default is 0

    % generating height data for rough surface

    w = exp(-2*([-M:1:M]*dx).^2/H_corr^2)';

    % weight function. The integration to obtain this exponent is in my notes

    normw = w/sum(w);  %here, dividing the w vector of weights by their sum - normalisation

    nn = (floor((round((Length/dx)*10)/10))+1); % now want to determine N having chosen L and 
                                                % evaluated M using Ogilvy's
                                                % formula
                                                % nn is 2L+1
    Numelvi = nn + 2*M;  % the number of uncorrelated random data
                         % Recall N = L + M
                         % so 2N+1 = 2L+2M+1

    RMS_v = sqrt(RMS_H^2/sum(normw.^2)); % the RMS value of uncorrelated random data v
                                         % equation (8) in the Ogilvy paper

    % s = (RandStream('mcg16807', 'Seed', seed));  
    % MATLAB sub-routine. Not sure about the best way to choose seed
    % this fixes the random set of numbers by using 'mcg16897'. If you want to
    % see different results for each realisation, you then have to change the
    % seed each time. 
 for kk = 1:numr
    v = RMS_v.*randn(Numelvi, 1);   % v vector of uncorrelated random numbers with 
                                    % approximately Gaussian distribution.
    while abs(mean(v))>0.01                                 
          v = RMS_v.*randn(Numelvi, 1);
    end
    checksig=(1/Numelvi)*sum(v.^2);  % ensemble averaging formula for v vector
    RMS_V_CHECK=sqrt(checksig);      % see equations (4) and (8) in Ogilvy paper.

    % the bigger numelvi is, obviously the closer RMS_v and RMS_V_CHECK are. The
    % latter value is using the ensemble averaging formula.
 

    for i = 1:nn
    z(i) = sum(normw.*v(i:1:i+2*M));  %see equation (5) in Ogilvy paper
    end
 
    


xaxis =-Length/2:dx:Length/2;


%{
figure
plot(xaxis,z);
%}


% compare with the theoretical Gaussian distribution
zstd=std(z);
% Only take a section of the surface to be the surface of interest ( if the
% surface length used for generation is not long enough, the statistics is
% not stable);

xcrack0=-Finitelength/2;
xcrack1=Finitelength/2;

Lsection = xcrack1-xcrack0;

idx_start = round(Length/2/dx)-round(Lsection/2/dx);
idx_end = round(Length/2/dx)+round(Lsection/2/dx);

xaxisS = -Lsection/2:dx:Lsection/2;
zs = z(idx_start:idx_end);

 
%{
figure;
plot(xaxisS,zs);
title('Rough surface');
axis equal
set(gca,'FontSize',12);
xlabel('length (m)','FontSize',14);
ylabel('height (m)','FontSize',14);
%}

% centre of crack:
xc=0;
zc=0;

%create the bottom surface of the finite crack
lx = max(xaxisS)-min(xaxisS);

%eps=0.01;
kx = pi/lx;
scale = 0.5e-3;


%*H_corr/Twavelength;
%offset = abs(2*min(z_w));
offset = abs(2*min(zs(5:end-5)));

if offset < 0.1
    offset=0.25;
    Botmaker = -2*scale*sin(kx*(xaxisS-min(xaxisS)))+1 ;
else
    Botmaker = -scale*sin(kx*(xaxisS-min(xaxisS)))+1 ;
end

offset;
    zb = (zs + offset).* Botmaker - offset; 

zbc = (zb(1)+zb(end))/2;

for ii = 1:length(xaxisS)
   zbn(ii) =  (zb(ii)-zbc) +zbc;
end

%{
figure;
plot(xaxisS,zs);
hold on
plot(xaxisS,zbn,'k')
axis equal
%}

% now have 2 sets of height data, representing an upper and lower crack with
% x-coordinates given by xaxisS, and zs representing upper heights, and zbn
% the bottom heights. We now want to rotate this data through an arbitrary
% angle using rotation matrices. First need to find centre of defect.
xcentre = 0e-3;
zcentre = 0.5*(zs((length(zs)+1)/2)+zbn((length(zs)+1)/2));
% plot(xcentre,zcentre,'rx');


% rotate the crack by angle alpha, which is between x and z axis,
% counter-clockwise
alphad=90;
alpha=pi*alphad/180;                               % rotation angle, anticlockwise
xaxisnewT = zeros(size(xaxisS));          % need x data
xaxisnewB = zeros(size(xaxisS)); 
zdatanewT = zeros(size(zs));
zdatanewB = zeros(size(zs));


   for ii = 1:length(xaxisS)
    xaxisnewT(ii) = cos(alpha)*(xaxisS(ii)-xcentre) - sin(alpha)*(zs(ii)-zcentre) + xcentre;
    xaxisnewB(ii) = cos(alpha)*(xaxisS(ii)-xcentre) - sin(alpha)*(zbn(ii)-zcentre) + xcentre;
    zdatanewT(ii) = sin(alpha)*(xaxisS(ii)-xcentre) + cos(alpha)*(zs(ii)-zcentre) + zcentre;
    zdatanewB(ii) = sin(alpha)*(xaxisS(ii)-xcentre) + cos(alpha)*(zbn(ii)-zcentre) + zcentre;
   end


% may want to comment this figure if producing several hundred/thousand
% surfaces
% figure
%     plot(xaxisS,zs,'b','LineWidth',2);
%     hold on
%     plot(xaxisS,zbn,'r');
%     %plot(xaxisnewT,zdatanewT,'k');
%     %plot(xaxisnewB,zdatanewB,'r','LineStyle','--');
%     axis equal




% % %save(['1DsurfaceFLAT15mm_', num2str(alphad), '.mat'], 'xaxisnewT','xaxisnewB','zdatanewT','zdatanewB');

% save(['/Users/stewarthaslinger/Documents/ML_TOFD_Corcoran/roughprofiles_raw/1DsurfaceRS_6mm_', ...
% num2str(alphad),'_',num2str(Hco),'_',num2str(RSh),'_',num2str(kk), '.mat'], 'xaxisnewT','xaxisnewB','zdatanewT','zdatanewB');
D = [xaxisnewT;xaxisnewB];
D = transpose(D);
 end
end