function features_matrix = generate_features(ecgs, Fs)
    if nargin < 2
        Fs = 300;
    end
    features_matrix = zeros(size(ecgs,1),58);
    for i=1:size(ecgs,1)
        features_matrix(i,:) = cell2mat(struct2cell(get_features(ecgs{i}, Fs)));
        %complex = features_matrix(i,:);
        %complex(~isreal(complex)) = -arrayfun(@imag, complex(~isreal(complex)));
        %features_matrix(i,:) = arrayfun(@real, complex);
    end
    features_matrix = fillmissing(features_matrix,'knn',5);
    features_matrix = normalize(features_matrix);
end


function features = get_features(ecgSignal, Fs)
    % ECG Feature Extraction
    %Inputs:
    % ecgSignal: ECG signal vector
    % fs: Sampling frequency
    %Outputs:
    % features matrix
    
    %NaN values
    ecgSignal(isnan(ecgSignal))=0;

    %Filtering and feature exraction
    filtered_ecg = Sanghavi_filtering(ecgSignal);
    
    %Feature selection
    features = struct();
    
    features.MeanSig = mean(filtered_ecg);
    features.StdSig = std(filtered_ecg);
    features.SkewSig = skewness(filtered_ecg);
    features.KurtSig = kurtosis(filtered_ecg);
    features.EntSig = entropy(filtered_ecg);

    % Pan-Tompkins Algorithm for QRS Detection
    [~,R_locs] = pan_tompkin(ecgSignal, Fs);
    % RR Interval Calculation
    RR_intervals = diff(R_locs);

    % Statistical Feature Calculation
    features.HeartRate = length(R_locs) / (length(ecgSignal));
    features.MeanRR = mean(RR_intervals);
    features.StdRR = std(RR_intervals);
    features.SkewRR = skewness(RR_intervals);
    features.KurtRR = kurtosis(RR_intervals);
    
    RR_intervals(RR_intervals==0) = NaN;
    log_RR_intervals = log(RR_intervals);
    
    features.MeanLogRR = mean(log_RR_intervals);
    features.StdLogRR = std(log_RR_intervals);
    features.SkewLogRR = skewness(log_RR_intervals);
    features.KurtLogRR = kurtosis(log_RR_intervals);

    % the rest of P-QRS-T complex
    [Q_locs, S_locs, P_locs, T_locs] = PQRST_complex(ecgSignal, R_locs, Fs);
    %QT PR QR SS ST RS
    % P Q R S T
    QT_intervals = T_locs' - Q_locs';
    PR_intervals = P_locs' - R_locs;
    SS_intervals = diff(S_locs');
    ST_intervals = T_locs' - S_locs';
    RS_intervals = S_locs' - R_locs;
    %Calculate statistics
    features.MeanQT = mean(QT_intervals);
    features.StdQT = std(QT_intervals);
    features.SkewQT = skewness(QT_intervals);
    features.KurtQT = kurtosis(QT_intervals);

    features.MeanPR = mean(PR_intervals);
    features.StdPR = std(PR_intervals);
    features.SkewPR = skewness(PR_intervals);
    features.KurtPR = kurtosis(PR_intervals);

    features.MeanSS = mean(SS_intervals);
    features.StdSS = std(SS_intervals);
    features.SkewSS = skewness(SS_intervals);
    features.KurtSS = kurtosis(SS_intervals);

    features.MeanST = mean(ST_intervals);
    features.StdST = std(ST_intervals);
    features.SkewST = skewness(ST_intervals);
    features.KurtST = kurtosis(ST_intervals);

    features.MeanRS = mean(RS_intervals);
    features.StdRS = std(RS_intervals);
    features.SkewRS = skewness(RS_intervals);
    features.KurtRS = kurtosis(RS_intervals);
    
    %Normalized
    ecgSignal = normalize(filtered_ecg);
    % Daubechies Wavelet Transform
    [C,L] = wavedec(ecgSignal, 10, 'db6');
    % put a7 & d7 to 0 in order to get the QRS complex
    % keep d1-6
    C(1:L(1)+L(2)+L(3)+L(4)+L(5)) = 0;
    db6 = waverec(C,L,'db6');
    [C,L] = wavedec(ecgSignal, 10, 'sym4');
    C(1:L(1)+L(2)+L(3)+L(4)+L(5)) = 0;
    sym4 = waverec(C,L,'sym4');
    [C,L] = wavedec(ecgSignal, 10, 'bior3.1');
    C(1:L(1)+L(2)+L(3)+L(4)+L(5)) = 0;
    bior3 = waverec(C,L,'bior3.1');

    %[~, cD3] = dwt(ecgSignal, 'db3');
    %[~, cD4] = dwt(ecgSignal, 'db4');
    %[~, cD5] = dwt(ecgSignal, 'db5');

    % Wavelet Feature Extraction
    features.MeanWavDetail3 = mean(db6);
    features.MaxWavDetail3 = max(db6);
    features.VarWavDetail3 = var(db6);
    features.EntWavDetail3 = entropy(db6);

    features.MeanWavDetail4 = mean(sym4);
    features.MaxWavDetail4 = max(sym4);
    features.VarWavDetail4 = var(sym4);
    features.EntWavDetail4 = entropy(sym4);

    features.MeanWavDetail5 = mean(bior3);
    features.MaxWavDetail5 = max(bior3);
    features.VarWavDetail5 = var(bior3);
    features.EntWavDetail5 = entropy(bior3);
    %entropy
    [bl,al] = butter(4, [0.01, 3]*2/Fs,'bandpass');
    ecg03 = filter(bl,al,ecgSignal);
    [bl,al] = butter(4, [3, 10]*2/Fs,'bandpass');
    ecg310 = filter(bl,al,ecgSignal);
    [bl,al] = butter(4, [10, 20]*2/Fs,'bandpass');
    ecg1020 = filter(bl,al,ecgSignal);
    
    features.MeanFilt03 = mean(ecg03);
    features.MaxFilt03 = max(ecg03);
    features.VarFilt03 = var(ecg03);
    features.EntFilt03 = entropy(ecg03);

    features.MeanFilt310 = mean(ecg310);
    features.MaxFilt310 = max(ecg310);
    features.VarFilt310 = var(ecg310);
    features.EntFilt310 = entropy(ecg310);

    features.MeanFilt1020 = mean(ecg1020);
    features.MaxFilt1020 = max(ecg1020);
    features.VarFilt1020 = var(ecg1020);
    features.EntFilt1020 = entropy(ecg1020);

end
%%
function [r_peaks, adjusted_locs] = pan_tompkin(ecg_signal, Fs)
    % Bandpass filter for ECG signal
    % fs is the sampling frequency
    Wp = [12 30]/(Fs/2); % passband
    Ws = [1 50]/(Fs/2); % stopband
    [n,Wn] = buttord(Wp,Ws,3,20); % order and natural frequency
    [b,a] = butter(n,Wn); % filter coefficients
    filtered_ecg = filtfilt(b, a, ecg_signal); % zero-phase filtering
    differentiation_filter = [1, 2, 0, -2, -1] * (Fs/8);
    differentiated_ecg = conv(filtered_ecg, differentiation_filter, 'same');
    squared_ecg = differentiated_ecg .^ 2;
    N = round(0.1 * Fs); % 150 ms window length
    integration_window = ones(1, N) / N;
    integrated_ecg = conv(squared_ecg, integration_window, 'same');
    threshold = mean(integrated_ecg) + 0.1*sqrt(var(integrated_ecg));  % Example threshold;
    [~, locs] = findpeaks(integrated_ecg, 'MinPeakHeight', threshold,'MinPeakDistance',0.4*Fs);
    % Adjust QRS locations by finding nearest maxima in original ECG
    adjusted_locs = zeros(size(locs));
    
    % Parameters
    search_radius = 20; % Define the radius around the detected peak to search for the true maxima
    
    for i = 1:length(locs)
        % Define search window around detected QRS location
        window_start = max(1, locs(i) - search_radius);
        window_end = min(length(ecg_signal), locs(i) + search_radius);
        search_window = window_start:window_end;
    
        % Find the index of the maximum value in the search window
        [~, max_idx] = max(ecg_signal(search_window));
        
        % Adjust location to the index of the maximum value
        adjusted_locs(i) = window_start + max_idx - 1;
    end
    r_peaks = ecg_signal(adjusted_locs);
end

function [Q_locs, S_locs, P_locs, T_locs] = PQRST_complex(ecg_signal, R_locs,Fs)
    % Initialize arrays to store the locations of P, Q, S, and T waves
    P_locs = [];
    Q_locs = [];
    S_locs = [];
    T_locs = [];

    % Define search windows (in seconds)
    window_QS = 0.05; % 50 milliseconds around R peak for Q and S
    window_P = 0.2;   % 200 milliseconds before Q for P
    window_T = 0.2;   % 200 milliseconds after S for T

    for i = 1:length(R_locs)
        R = R_locs(i);

        % Search for Q and S
        start_QS = max(1, R - round(window_QS * Fs));
        end_QS = min(length(ecg_signal), R + round(window_QS * Fs));
        [~, Q_idx] = min(ecg_signal(start_QS:R)); % Q wave is a minimum before R
        [~, S_idx] = min(ecg_signal(R:end_QS));   % S wave is a minimum after R
        Q_locs = [Q_locs, start_QS + Q_idx - 1];
        S_locs = [S_locs, R + S_idx - 1];

        % Search for P wave
        % Assuming P wave is before Q wave
        start_P = max(1, Q_locs(end) - round(window_P * Fs));
        [~, P_idx] = max(ecg_signal(start_P:Q_locs(end))); % P wave is a peak before Q
        P_locs = [P_locs, start_P + P_idx - 1];

        % Search for T wave
        % Assuming T wave is after S wave
        end_T = min(length(ecg_signal), S_locs(end) + round(window_T * Fs));
        [~, T_idx] = max(ecg_signal(S_locs(end):end_T)); % T wave is a peak after S
        T_locs = [T_locs, S_locs(end) + T_idx - 1];
    end
    % Q_peaks = ecg_signal(Q_locs);
    % S_peaks = ecg_signal(S_locs);
    % P_peaks = ecg_signal(P_locs);
    % T_peaks = ecg_signal(T_locs);
end

function filtered_ecg = Sanghavi_filtering(ecg_signal)
    [C,L] = wavedec(ecg_signal, 10, 'db4');
    threshold = 0.15;  % Define your threshold
    C(abs(C) < threshold) = 0;
    filtered_ecg = waverec(C,L,'db4');
    % Moving average filter
    windowSize = 10;  
    maFilter = ones(1, windowSize) / windowSize;
    filtered_ecg = conv(filtered_ecg, maFilter, 'same');
    %Savitzky-Golay filter
    order = 15;
    framelen = 29;
    filtered_ecg = sgolayfilt(filtered_ecg,order,framelen);
end