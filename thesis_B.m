function analyze_stageAB()
% Analyze Stage A (Python-only) and Stage B (MuJoCo) CSVs and produce
% all figures + tables referenced by the LaTeX excerpts.
%
% Stage A expects (current directory):
%   PID-Light.csv,  MPC-Light.csv,  POMDP-Light.csv
%   PID-Nominal.csv,MPC-Nominal.csv,POMDP-Nominal.csv
%   PID-Severe.csv, MPC-Severe.csv, POMDP-Severe.csv
%
% Stage B expects (current directory):
%   PID-saw.csv, PID-sin.csv, PID-hh.csv
%   MPC-saw.csv, MPC-sin.csv, MPC-hh.csv
%   POMDP-saw.csv, POMDP-sin.csv, POMDP-hh.csv
%
% Outputs:
%   Stage A:
%     results/python_sim/time_work_tradeoff_python.png
%     results/python_sim/state_traces_python.png
%     stageA_summary.csv
%     stageA_summary.tex
%   Stage B:
%     friction_spectrum_grid.png
%     time_work_box_mujoco.png
%     rank_correlation_stageA_stageB.png
%     stageB_summary.csv
%     stageB_summary.tex

fprintf('=== Stage A (Python-only) ===\n');
SA = analyze_stageA_python();

fprintf('\n=== Stage B (MuJoCo) ===\n');
SB = analyze_stageB_mujoco(SA);  %#ok<NASGU>
fprintf('\nAll outputs written.\n');
end


%% ========================== Stage A (Python-only) ==========================
function SA = analyze_stageA_python()
% Builds Stage-A (Group A) summary and figures from per-trace CSVs.

% ----------------------------- Configuration -----------------------------
DATA_DIR   = ".";                         % current directory
OUT_DIR    = fullfile("results","python_sim");
if ~exist(OUT_DIR, "dir"), mkdir(OUT_DIR); end

ESC_TOL_DEG = 2.0;                        % ±2 deg band
ESC_DWELL_S = 0.50;                       % dwell time
TAU_SAT     = 5.0;                        % N·m (match Python)
SAT_FRAC    = 0.98;                       % saturation threshold
REL_CLOSE_PCT = 0.04;                     % 4% fallback rule
NZ_EPS        = 1e-9;

PROFILES = struct('key',{'Light','Nominal','Severe'}, 'name',{'Light','Nominal','Severe'});
CTRLS    = struct('key',{'PID','MPC','POMDP'},        'name',{'PID','MPC/MPPI','POMDP'});

% ---------------------------- Ingest & Metrics ----------------------------
files = dir(fullfile(DATA_DIR, "*.csv"));
names = string({files.name});

metrics = [];  % per-file rows for bootstrapping/selection of reps
for ip = 1:numel(PROFILES)
    prof = PROFILES(ip).key;
    for ic = 1:numel(CTRLS)
        ctrl = CTRLS(ic).key;

        % match files: '<CTRL>-<PROFILE>*.csv' (insensitive). Excludes Stage‑B families.
        patt = lower(ctrl) + "-" + lower(prof);
        cand = names(contains(lower(names), patt));

        for k = 1:numel(cand)
            T = readtable(cand(k), "TextType","string", "NumHeaderLines",0);

            t         = getVar(T, "t", []);
            angle     = getVar(T, "angle", []);
            setpoint  = getVar(T, "setpoint", []);
            sp_goal   = getVar(T, "sp_goal", []);
            dq        = getVar(T, "dq", []);
            tau_cmd   = getVar(T, "tau_cmd", []);

            if isempty(t) || isempty(angle) || (isempty(setpoint) && isempty(sp_goal)) || isempty(dq) || isempty(tau_cmd)
                continue;  % skip malformed
            end

            % choose target (prefer sp_goal if present)
            if ~isempty(sp_goal), target = last_finite(sp_goal);
            else,                 target = last_finite(setpoint);
            end

            esc_tol = deg2rad(ESC_TOL_DEG);
            t_esc = escape_time(t, angle, target, esc_tol, ESC_DWELL_S);

            if ~isfinite(t_esc)
                % fallback 4% + non-zero
                if isempty(setpoint) && ~isempty(sp_goal), setpoint = sp_goal; end
                if isempty(sp_goal)  && ~isempty(setpoint), sp_goal  = setpoint; end
                t_fb = fallback_tesc_rel4pct(t, angle, setpoint, sp_goal, REL_CLOSE_PCT, NZ_EPS);
                if isfinite(t_fb), t_esc = t_fb; end
            end

            W_mech = trapz(t, abs(tau_cmd .* dq));
            dirSign = sign(target - angle(1));
            overs = dirSign .* (angle - target);
            theta_os = max([0; overs(:)]);
            sat_mask = abs(tau_cmd) >= SAT_FRAC * TAU_SAT;
            sat_ms   = 1000 * sum(sample_dwell_t(t, sat_mask));

            r = struct();
            r.Profile     = PROFILES(ip).name;
            r.Controller  = CTRLS(ic).name;
            r.file        = cand(k);
            r.t_esc_s     = t_esc;
            r.W_mech_J    = W_mech;
            r.theta_os_rad= theta_os;
            r.Sat_time_ms = sat_ms;
            metrics = [metrics; r]; %#ok<AGROW>
        end
    end
end

if isempty(metrics)
    error('Stage A: no usable CSVs found (expected PID/MPC/POMDP × Light/Nominal/Severe).');
end

M = struct2table(metrics);
M.Profile    = string(M.Profile);
M.Controller = string(M.Controller);

% ----------------------------- Medians & Boots -----------------------------
profs = unique(M.Profile,'stable');
ctrls = unique(M.Controller,'stable');

Summary = table('Size',[0 6], ...
    'VariableTypes', {'string','string','double','double','double','double'}, ...
    'VariableNames', {'Profile','Controller','t_esc_s','W_mech_J','theta_os_rad','Sat_time_ms'});

Boots = struct();  % Boots.(ctrl).(prof) -> (B×2) matrix of boot medians [t_esc, W_mech]
B = 600;

for ip = 1:numel(profs)
    for ic = 1:numel(ctrls)
        rows = M(M.Profile==profs(ip) & M.Controller==ctrls(ic), :);
        if isempty(rows), continue; end

        med_t = median(rows.t_esc_s, 'omitnan');
        med_w = median(rows.W_mech_J, 'omitnan');
        med_o = median(rows.theta_os_rad, 'omitnan');
        med_s = median(rows.Sat_time_ms, 'omitnan');

        Summary = [Summary; table(profs(ip), ctrls(ic), med_t, med_w, med_o, med_s, ...
            'VariableNames', Summary.Properties.VariableNames)]; %#ok<AGROW>

        % bootstrap medians over trials
        vals = table(rows.t_esc_s, rows.W_mech_J, 'VariableNames',{'t','w'});
        if height(vals) >= 2
            boot = zeros(B,2);
            idxs = (1:height(vals))';
            for b = 1:B
                ii = idxs( randi(numel(idxs), numel(idxs), 1) );
                boot(b,1) = median(vals.t(ii),'omitnan');
                boot(b,2) = median(vals.w(ii),'omitnan');
            end
        else
            boot = [med_t, med_w];  % degenerate: single row
        end
        Boots.(matKey(ctrls(ic))).(matKey(profs(ip))) = boot;
    end
end

% ------------------------------ Save summary -------------------------------
writetable(Summary, fullfile(".", "stageA_summary.csv"));
writeLatexStageA(Summary, fullfile(".", "stageA_summary.tex"));
fprintf('Stage A: wrote %s and %s\n', fullfile(".", "stageA_summary.csv"), fullfile(".", "stageA_summary.tex"));

% ------------------------ Figure: time–work tradeoff ------------------------
fig = figure('Name','Stage A time–work','Color','w','Position',[160 160 760 560]);
hold on; grid on; box on;
colors = containers.Map({'PID','MPC/MPPI','POMDP'}, ...
    {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.4660 0.6740 0.1880]});
markers= containers.Map({'Light','Nominal','Severe'}, {'o','s','^'});

% scatter points + 95% bootstrap ellipses
for r = 1:height(Summary)
    c = Summary.Controller(r);
    p = Summary.Profile(r);
    plot(Summary.t_esc_s(r), Summary.W_mech_J(r), markers(char(p)), 'MarkerFaceColor','none', ...
        'MarkerEdgeColor', colors(char(c)), 'LineWidth',1.6, 'MarkerSize',7);
    boot = Boots.(matKey(c)).(matKey(p));
    mu = [Summary.t_esc_s(r), Summary.W_mech_J(r)];
    if size(boot,1) >= 2
        C = cov(boot,1); % population covariance
        pts = ellipse_points(mu, C, 180);
        plot(pts(:,1), pts(:,2), '-', 'Color', colors(char(c)), 'LineWidth',1.0, 'HandleVisibility','off');
    end
end
xlabel('$t_{\mathrm{esc}}$ (s)', 'Interpreter','latex');
ylabel('$W_{\mathrm{mech}}$ (J)', 'Interpreter','latex');
title('Stage A (Python): time–work trade-off (medians + 95% boot CIs)');

% Legend for controller colors (clean & robust)
lc = ["PID","MPC/MPPI","POMDP"];
h_color = gobjects(1, numel(lc));
for i=1:numel(lc)
    h_color(i) = plot(nan,nan,'-','Color',colors(char(lc(i))),'LineWidth',1.5);
end
leg = legend(h_color, cellstr(lc), 'Location','northeast');
leg.Title.String = 'Controller';
legend('boxoff');

% Marker cue for profiles (compact text box)
annotation('textbox',[0.74 0.20 0.22 0.15],'String', ...
    sprintf('Markers:\\newline o  Light\\newline s  Nominal\\newline ^  Severe'), ...
    'EdgeColor','none','FontSize',10);

f1 = fullfile(OUT_DIR,'time_work_tradeoff_python.png');
set(fig,'PaperPositionMode','auto'); print(fig, f1, '-dpng', '-r300');
close(fig);
fprintf('Stage A: wrote %s\n', f1);

% ------------------- Figure: representative traces (Nominal) ----------------
fig2 = figure('Name','Stage A traces (Nominal)','Color','w','Position',[200 120 860 580]);
axs = [subplot(3,1,1), subplot(3,1,2), subplot(3,1,3)];
hold(axs(1),'on'); hold(axs(2),'on'); hold(axs(3),'on');

tmax = 0;
for ic = 1:numel(ctrls)
    c = ctrls(ic);
    rows_c = M(M.Profile=="Nominal" & M.Controller==c, :);
    if isempty(rows_c), continue; end
    med_tesc = median(rows_c.t_esc_s, 'omitnan');
    [~, idx] = min(abs(rows_c.t_esc_s - med_tesc));
    repFile = rows_c.file(idx);
    T = readtable(repFile, "TextType","string", "NumHeaderLines",0);
    t  = getVar(T, "t", []);
    ang= getVar(T, "angle", []);
    sp = getVar(T, "setpoint", []);
    dq = getVar(T, "dq", []);
    tu = getVar(T, "tau_cmd", []);

    if isempty(t) || isempty(ang) || isempty(sp) || isempty(dq) || isempty(tu), continue; end
    tmax = max(tmax, max(t));

    col = colors(char(c));
    plot(axs(1), t, ang, '-', 'Color', col, 'DisplayName', char(c)); 
    plot(axs(1), t, sp,  '--', 'Color', [0 0 0 0.55], 'HandleVisibility','off');

    plot(axs(2), t, dq, '-', 'Color', col, 'DisplayName', char(c));
    plot(axs(3), t, tu, '-', 'Color', col, 'DisplayName', char(c));
end

ylabel(axs(1), '\theta (rad)'); ylabel(axs(2), '\omega (rad/s)'); ylabel(axs(3), '\tau (N·m)');
xlabel(axs(3), 'time (s)');
title(axs(1), 'Stage A (Python): representative \theta(t), \omega(t), \tau(t) under Nominal');
for a = axs, grid(a,'on'); xlim(a,[0 tmax]); end
legend(axs(1),'Location','best');

f2 = fullfile(OUT_DIR,'state_traces_python.png');
set(fig2,'PaperPositionMode','auto'); print(fig2, f2, '-dpng', '-r300');
close(fig2);
fprintf('Stage A: wrote %s\n', f2);

SA = Summary;  % return for Stage‑B rank correlation
end


%% ========================= Stage B (MuJoCo) ===============================
function SB = analyze_stageB_mujoco(SA_from_A)
% Analyze Stage B (MuJoCo) CSVs and produce figures + summary table.
% If Stage‑A summary is provided, uses it for improved rank‑correlation figure.

DATA_DIR   = ".";              % CSVs live here
OUT_DIR    = ".";              % save everything to current directory
ESC_TOL_DEG = 2.0;             % ±2 deg band
ESC_DWELL_S = 0.50;            % dwell time
TAU_SAT     = 5.0;             % N·m
SAT_FRAC    = 0.98;            % saturation threshold

REL_CLOSE_PCT = 0.04;          % 4% fallback rule
NZ_EPS        = 1e-9;

FAMILIES = struct('key',  {'saw','sin','hh'}, ...
                  'name', {'Sawtooth','Sinusoidal','Multi-well'});
CTRLS    = struct('key',  {'PID','MPC','POMDP'}, ...
                  'name', {'PID','MPC/MPPI','POMDP'});

% Build expected file specs
specs = {};
for f = 1:numel(FAMILIES)
    fam = FAMILIES(f).key;
    for c = 1:numel(CTRLS)
        ctrl = CTRLS(c).key;
        fname = sprintf('%s-%s.csv', ctrl, fam);
        specs{end+1} = struct('family_key', fam, 'family_name', FAMILIES(f).name, ...
                              'ctrl_key', ctrl, 'ctrl_name', CTRLS(c).name, ...
                              'file', fullfile(DATA_DIR, fname)); %#ok<AGROW>
    end
end

% ----------------------------- Ingest + Metrics ---------------------------
records = [];
presentMap  = false(numel(FAMILIES), numel(CTRLS));
hasEscape   = false(numel(FAMILIES), numel(CTRLS));

for i = 1:numel(specs)
    S = specs{i};
    fi = find(strcmp({FAMILIES.key}, S.family_key));
    cj = find(strcmp({CTRLS.key},    S.ctrl_key));
    if ~isfile(S.file), continue; end
    presentMap(fi, cj) = true;

    T = readtable(S.file, "TextType","string", "NumHeaderLines",0);

    t         = getVar(T, "t", []);
    angle     = getVar(T, "angle", []);
    setpoint  = getVar(T, "setpoint", []);
    sp_goal   = getVar(T, "sp_goal", []);
    dq        = getVar(T, "dq", []);
    tau_cmd   = getVar(T, "tau_cmd", []);
    risk      = getVar(T, "risk", []);
    risk_act  = getVar(T, "risk_act", []);
    risk_err  = getVar(T, "risk_err", []);
    tau_load  = getVar(T, "tau_load", []);

    if isempty(t) || isempty(angle) || (isempty(setpoint) && isempty(sp_goal)) || isempty(dq) || isempty(tau_cmd)
        continue;
    end

    % Choose final target (prefer sp_goal)
    if ~isempty(sp_goal), target = last_finite(sp_goal);
    else,                 target = last_finite(setpoint);
    end

    esc_tol = deg2rad(ESC_TOL_DEG);
    t_esc = escape_time(t, angle, target, esc_tol, ESC_DWELL_S);

    if ~isfinite(t_esc)
        if isempty(setpoint) && ~isempty(sp_goal), setpoint = sp_goal; end
        if isempty(sp_goal)  && ~isempty(setpoint), sp_goal  = setpoint; end
        t_fb = fallback_tesc_rel4pct(t, angle, setpoint, sp_goal, REL_CLOSE_PCT, NZ_EPS);
        if isfinite(t_fb), t_esc = t_fb; end
    end

    W_mech = trapz(t, abs(tau_cmd .* dq));  % Joules
    dir    = sign(target - angle(1));
    overs  = dir .* (angle - target);
    theta_os = max([0; overs(:)]);
    sat_mask = abs(tau_cmd) >= SAT_FRAC * TAU_SAT;
    sat_time_ms = 1000 * sum(sample_dwell_t(t, sat_mask));

    trial = getVar(T, "trial", []);
    if isempty(trial), N_B = 1; else, N_B = numel(unique(trial(~isnan(trial)))); end

    rec = struct();
    rec.family_key  = S.family_key;
    rec.family_name = S.family_name;
    rec.ctrl_key    = S.ctrl_key;
    rec.ctrl_name   = S.ctrl_name;
    rec.file        = S.file;
    rec.t_esc       = t_esc;
    rec.W_mech      = W_mech;
    rec.theta_os    = theta_os;
    rec.sat_ms      = sat_time_ms;
    rec.N_B         = N_B;
    rec.t = t; rec.angle = angle; rec.target = target;
    rec.dq = dq; rec.tau_cmd = tau_cmd; rec.risk = risk; rec.risk_act = risk_act; rec.risk_err = risk_err; rec.tau_load = tau_load;

    records = [records; rec]; %#ok<AGROW>
    hasEscape(fi, cj) = isfinite(t_esc);
end

if isempty(records)
    error('Stage B: no usable CSVs found.');
end

% --------------------------- Build Summary Table ---------------------------
[~, fam_order]  = ismember({records.family_key}', {FAMILIES.key});
[~, ctrl_order] = ismember({records.ctrl_key}',   {CTRLS.key});
order_idx = sortrows([(1:numel(records))', fam_order, ctrl_order], [2 3]);
records = records(order_idx(:,1));

summary = table( ...
    string({records.family_name}'), string({records.ctrl_name}'), ...
    [records.t_esc]', [records.W_mech]', [records.theta_os]', [records.sat_ms]', [records.N_B]', ...
    'VariableNames', {'Family','Controller','t_esc_s','W_mech_J','theta_os_rad','Sat_time_ms','N_B'});

writetable(summary, fullfile(OUT_DIR, "stageB_summary.csv"));
writeLatexStageB(summary, fullfile(OUT_DIR, "stageB_summary.tex"));
fprintf('Stage B: wrote %s and %s\n', fullfile(OUT_DIR,"stageB_summary.csv"), fullfile(OUT_DIR,"stageB_summary.tex"));

% ------------------------------- Figure 1 -----------------------------------
families = {FAMILIES.name};
controllers = {CTRLS.name};

M_t = nan(numel(families), numel(controllers));
M_w = nan(numel(families), numel(controllers));
M_s = nan(numel(families), numel(controllers));
for r = 1:height(summary)
    i = find(strcmp(summary.Family{r}, families));
    j = find(strcmp(summary.Controller{r}, controllers));
    M_t(i,j) = summary.t_esc_s(r);
    M_w(i,j) = summary.W_mech_J(r);
    M_s(i,j) = summary.Sat_time_ms(r);
end

normMM = @(X) (X - min(X(:)))./max(eps, (max(X(:))-min(X(:))));
Nt = normMM(M_t); Nw = normMM(M_w); Ns = normMM(M_s);
DI = 0.5*Nt + 0.4*Nw + 0.1*Ns;

fig1 = figure('Name','Stage B Spectrum','Color','w','Position',[200 200 900 400]);
imagesc(DI); axis equal tight;
set(gca,'XTick',1:numel(controllers),'XTickLabel',controllers, ...
        'YTick',1:numel(families),'YTickLabel',families,'TickDir','out','FontSize',11);
cb = colorbar; cb.Label.String = 'Difficulty index (↑ harder)';
title('Stage B (MuJoCo): Friction spectrum proxy across controllers');
grid on;
saveas(fig1, fullfile(OUT_DIR, 'friction_spectrum_grid.png'));
close(fig1);
fprintf('Stage B: wrote %s\n', fullfile(OUT_DIR,'friction_spectrum_grid.png'));

% ------------------------------- Figure 2 -----------------------------------
[~, fam_idx] = ismember(summary.Family, families);
[~, ctl_idx] = ismember(summary.Controller, controllers);
nF = numel(families); nC = numel(controllers);
Tesc = nan(nF, nC); Wmec = nan(nF, nC);
for k = 1:height(summary)
    Tesc(fam_idx(k), ctl_idx(k)) = summary.t_esc_s(k);
    Wmec(fam_idx(k), ctl_idx(k)) = summary.W_mech_J(k);
end

fig2 = figure('Name','Stage B Time & Work','Color','w','Position',[200 200 1100 420]);

subplot(1,2,1);
b = bar(Tesc,'grouped'); grid on; box on;
set(gca,'XTick',1:nF,'XTickLabel',families,'FontSize',11);
legend(controllers,'Location','northwest'); ylabel('t_{esc} (s)');
title('Stage B: Escape time   (N/E = no escape; M = missing)');
yTop = max(Tesc(:),[],'omitnan'); if isempty(yTop)||~isfinite(yTop)||yTop<=0, yTop=1; end
ylim([0, yTop*1.25]); yAnno = yTop*1.05;

xpos = nan(nF, nC);
try
    for j = 1:nC
        xs = b(j).XEndPoints; xs = xs(:);
        xpos(1:numel(xs), j) = xs;
    end
catch
    x = 1:nF; gw = min(0.8, nC/(nC+1.5));
    for j = 1:nC
        xpos(:,j) = x - gw/2 + (2*j-1)*gw/(2*nC);
    end
end

for iF = 1:nF
    for jC = 1:nC
        if ~presentMap(iF,jC)
            if isfinite(xpos(iF,jC))
                text(xpos(iF,jC), yAnno, 'M','HorizontalAlignment','center', ...
                     'Color',[0.5 0 0],'FontWeight','bold');
            end
        elseif presentMap(iF,jC) && ~hasEscape(iF,jC)
            if isfinite(xpos(iF,jC))
                text(xpos(iF,jC), yAnno, 'N/E','HorizontalAlignment','center', ...
                     'Color',[0.85 0.33 0.10],'FontWeight','bold');
            end
        end
    end
end

subplot(1,2,2);
bar(Wmec,'grouped'); grid on; box on;
set(gca,'XTick',1:nF,'XTickLabel',families,'FontSize',11);
legend(controllers,'Location','northwest'); ylabel('W_{mech} (J)');
title('Stage B: Mechanical work');

saveas(fig2, fullfile(OUT_DIR, 'time_work_box_mujoco.png'));
close(fig2);
fprintf('Stage B: wrote %s\n', fullfile(OUT_DIR,'time_work_box_mujoco.png'));

% ----------------------- Figure 3: A→B agreement (improved) ----------------
fig3 = figure('Name','Stage A→B agreement','Color','w','Position',[220 220 1150 420]);

if nargin>=1 && ~isempty(SA_from_A) && any(strcmp('Controller', SA_from_A.Properties.VariableNames))
    fams = unique(summary.Family,'stable');       % e.g., {'Sawtooth','Sinusoidal','Multi-well'}
    controllersAll = {CTRLS.name};                % {'PID','MPC/MPPI','POMDP'}

    % Stage A medians by controller (across Light/Nominal/Severe)
    SAc = groupsummary(SA_from_A, 'Controller', 'median', 't_esc_s');
    % Align Stage A vector to our canonical controller order
    [~, ia] = ismember(controllersAll, cellstr(SAc.Controller));
    SA_vec = nan(numel(controllersAll),1);
    okA = ia>0;
    SA_vec(okA) = SAc.median_t_esc_s(ia(okA));

    % Layout: left tile = bars of ρ per family; right tiles = mini-scatters
    tlay = tiledlayout(fig3, 1, numel(fams)+1, 'TileSpacing','compact', 'Padding','compact');

    % --- Leftmost tile: bars of Spearman ρ per family with p-values ---
    axBar = nexttile(tlay, 1);
    hold(axBar,'on'); grid(axBar,'on'); box(axBar,'on');
    R = nan(numel(fams),1); P = nan(numel(fams),1); N = zeros(numel(fams),1);

    for i = 1:numel(fams)
        SBf = summary(strcmp(summary.Family, fams{i}), :);              % this family only
        [~, ib] = ismember(controllersAll, cellstr(SBf.Controller));    % align by controller
        goodB = ib>0;

        A = SA_vec(goodB);
        B = SBf.t_esc_s(ib(goodB));
        m = isfinite(A) & isfinite(B);
        A = A(m); B = B(m);
        N(i) = numel(A);

        if N(i) >= 2
            R(i) = corr(A, B, 'Type','Spearman', 'Rows','pairwise');
            P(i) = spearman_perm_pval(A, B);                            % exact p-value for small n
        else
            R(i) = NaN; P(i) = NaN;
        end
    end

    bar(axBar, R, 'FaceColor', [0.65 0.72 0.92]);
    set(axBar, 'XTick', 1:numel(fams), 'XTickLabel', fams, 'FontSize', 11);
    ylim(axBar, [-1 1]); ylabel(axBar, 'Spearman \rho'); 
    title(axBar, 'Stage A \rightarrow B rank-correlation by family');

    % annotate ρ / p / n on each bar
    for i = 1:numel(fams)
        if isfinite(R(i))
            y = R(i); x = i;
            text(axBar, x, y + 0.08*sign(y + eps), ...
                 sprintf('\\rho=%.2f\\n(p=%.2f; n=%d)', R(i), P(i), N(i)), ...
                 'HorizontalAlignment','center','VerticalAlignment','bottom','FontSize',9);
        else
            text(axBar, i, 0, 'n<2', 'HorizontalAlignment','center', ...
                 'VerticalAlignment','middle','Color',[0.5 0 0]);
        end
    end

    % --- Right tiles: one mini-scatter per family (with identity line) ---
    for i = 1:numel(fams)
        ax = nexttile(tlay, i+1);
        hold(ax,'on'); grid(ax,'on'); box(ax,'on');

        SBf = summary(strcmp(summary.Family, fams{i}), :);
        [~, ib] = ismember(controllersAll, cellstr(SBf.Controller));
        goodB = ib>0;
        A = SA_vec(goodB);
        B = SBf.t_esc_s(ib(goodB));
        labels = controllersAll(goodB);

        m = isfinite(A) & isfinite(B);
        A = A(m); B = B(m); labels = labels(m);

        if isempty(A)
            text(ax, 0.5, 0.5, 'no overlap', 'Units','normalized', ...
                 'HorizontalAlignment','center'); axis(ax,'off'); continue;
        end

        scatter(ax, A, B, 36, 'filled');
        lo = min([A(:); B(:)]); hi = max([A(:); B(:)]);
        if ~isfinite(lo) || ~isfinite(hi) || lo==hi, lo=0; hi=1; end
        plot(ax, [lo hi], [lo hi], 'k--', 'LineWidth', 1);    % 45° line

        for k = 1:numel(A)
            text(ax, A(k), B(k), ['  ' labels{k}], 'FontSize', 9, 'VerticalAlignment','middle');
        end
        xlabel(ax, 'Stage A median t_{esc} (s)');
        ylabel(ax, sprintf('Stage B %s t_{esc} (s)', fams{i}));
        ttl = sprintf('%s (\\rho=%.2f)', fams{i}, R(i));
        title(ax, ttl);
    end

else
    % Fallback (B-only): correlation of controller ranks across families
    fams = unique(summary.Family);
    rho = nan(numel(fams));
    for i = 1:numel(fams)
        for j = 1:numel(fams)
            if i==j, rho(i,j) = 1; continue; end
            Si = summary(strcmp(summary.Family,fams{i}),:);
            Sj = summary(strcmp(summary.Family,fams{j}),:);
            [~, ia] = ismember(Si.Controller, Sj.Controller);
            ok = ia>0;
            if sum(ok) >= 2
                ai = Si.t_esc_s(ok);
                aj = Sj.t_esc_s(ia(ok));
                rho(i,j) = corr(ai, aj, 'Type','Spearman','Rows','pairwise');
            end
        end
    end
    imagesc(rho, [-1 1]); axis equal tight; colorbar; grid on;
    set(gca,'XTick',1:numel(fams),'XTickLabel',fams,'YTick',1:numel(fams),'YTickLabel',fams,'TickDir','out','FontSize',11);
    title('Stage B proxy: rank correlation across families');
end
print(fig3, fullfile(OUT_DIR,'rank_correlation_stageA_stageB.png'), '-dpng', '-r300');
close(fig3);
fprintf('Stage B: wrote %s\n', fullfile(OUT_DIR,'rank_correlation_stageA_stageB.png'));

SB = summary;
end


%% =============================== Utilities =================================
function v = getVar(T, name, defaultVal)
% Robustly fetch a variable from table T; return defaultVal if missing.
    if ismember(name, T.Properties.VariableNames)
        v = T.(name);
        if iscell(v); try v = cellfun(@str2double, v); catch, end, end
        if isstring(v); try v = double(v); catch, end, end
        v = v(:);
        if ~isnumeric(v) || all(~isfinite(v))
            v = defaultVal;
        end
    else
        v = defaultVal;
    end
end

function x = last_finite(v)
% Last finite value of a vector, else NaN
    idx = find(isfinite(v), 1, 'last');
    if isempty(idx), x = NaN; else, x = v(idx); end
end

function tEsc = escape_time(t, angle, target, tol, dwell)
% First time when |angle - target| <= tol and remains within tol for >= dwell
    if isempty(t) || isempty(angle) || ~isfinite(target)
        tEsc = NaN; return;
    end
    inside = abs(angle - target) <= tol;
    if ~any(inside), tEsc = NaN; return; end
    tEsc = NaN;
    i = 1; N = numel(t);
    while i <= N
        if ~inside(i)
            i = i + 1; continue;
        end
        j = i;
        while j <= N && inside(j)
            j = j + 1;
        end
        segTime = t(min(j-1,N)) - t(i);
        if segTime >= dwell
            tEsc = t(i); return;
        end
        i = j + 1;
    end
end

function tEsc = fallback_tesc_rel4pct(t, angle, setpoint, sp_goal, pct, nz_eps)
% Fallback escape time:
% First time where angle, setpoint, and sp_goal are ALL:
%   • finite and |value| > nz_eps (non-zero)
%   • mutually within 'pct' (4%) pairwise relative difference.
% Returns NaN if no such time exists.
    if isempty(t) || isempty(angle) || isempty(setpoint) || isempty(sp_goal)
        tEsc = NaN; return;
    end
    n = min([numel(t), numel(angle), numel(setpoint), numel(sp_goal)]);
    if n <= 0, tEsc = NaN; return; end
    t  = t(1:n);
    a  = angle(1:n);
    sp = setpoint(1:n);
    sg = sp_goal(1:n);

    finiteMask = isfinite(a) & isfinite(sp) & isfinite(sg);
    if ~any(finiteMask), tEsc = NaN; return; end
    a  = a(finiteMask);
    sp = sp(finiteMask);
    sg = sg(finiteMask);
    t  = t(finiteMask);

    nonzero = (abs(a) > nz_eps) & (abs(sp) > nz_eps) & (abs(sg) > nz_eps);

    % Pairwise relative closeness: max(|xi-xj|) <= pct * max(|xi|,|xj|,|xk|)
    maxv = max([abs(a), abs(sp), abs(sg)], [], 2);
    spread = max([abs(a - sp), abs(a - sg), abs(sp - sg)], [], 2);
    close = spread <= pct .* maxv;

    cond = nonzero & close;
    idx = find(cond, 1, 'first');
    if isempty(idx)
        tEsc = NaN;
    else
        tEsc = t(idx);
    end
end

function total = sample_dwell_t(t, mask)
% Sum of time spent where mask==true, using forward diffs
    if isempty(t) || isempty(mask) || numel(t)~=numel(mask)
        total = 0; return;
    end
    dt = [diff(t); median(diff(t))];
    total = sum(dt(mask));
end

function writeLatexStageA(S, outPath)
% Write LaTeX table for Stage A (profiles)
    fid = fopen(outPath,'w');
    if fid < 0
        warning('Could not write %s', outPath); return;
    end
    fprintf(fid, '%% Auto-generated by analyze_stageAB.m (Stage A)\n');
    fprintf(fid, '\\begin{table}[H]\n  \\centering\n');
    fprintf(fid, '  \\caption{Stage A summary}\n');
    fprintf(fid, '  \\label{tab:stageA_summary}\n  \\vspace{4pt}\n');
    fprintf(fid, '  \\begin{tabular}{l l c c c c}\n    \\toprule\n');
    fprintf(fid, '    Profile & Controller & $t_{\\mathrm{esc}}$ (s) & $W_{\\mathrm{mech}}$ (J) & $\\theta_{\\mathrm{os}}$ (rad) & Sat. time (ms)\\\\\n');
    fprintf(fid, '    \\midrule\n');

    profs = unique(S.Profile,'stable');
    for i = 1:numel(profs)
        prof = profs(i);
        rows = S(S.Profile==prof,:);
        for j = 1:height(rows)
            fmt = '    %s & %s & %.3f & %.3f & %.3f & %.0f \\\\\n';
            if j > 1, profLabel = ' '; else, profLabel = char(prof); end
            fprintf(fid, fmt, profLabel, rows.Controller(j), rows.t_esc_s(j), rows.W_mech_J(j), ...
                           rows.theta_os_rad(j), rows.Sat_time_ms(j));
        end
    end
    fprintf(fid, '    \\bottomrule\n  \\end{tabular}\n\\end{table}\n');
    fclose(fid);
end

function writeLatexStageB(S, outPath)
% Write LaTeX table matching Stage B excerpt layout
    fid = fopen(outPath,'w');
    if fid < 0
        warning('Could not write %s', outPath); return;
    end
    fprintf(fid, '%% Auto-generated by analyze_stageAB.m (Stage B)\n');
    fprintf(fid, '\\begin{table}[H]\n  \\centering\n');
    fprintf(fid, '  \\caption{Stage B summary across friction families.}\n');
    fprintf(fid, '  \\label{tab:stageB_summary}\n  \\vspace{4pt}\n');
    fprintf(fid, '  \\begin{tabular}{l l c c c c c}\n    \\toprule\n');
    fprintf(fid, '    Family & Controller & $t_{\\mathrm{esc}}$ (s) & $W_{\\mathrm{mech}}$ (J) & $\\theta_{\\mathrm{os}}$ (rad) & Sat. time (ms) & $N_\\text{B}$\\\\\n');
    fprintf(fid, '    \\midrule\n');

    fams = unique(S.Family,'stable');
    for i = 1:numel(fams)
        fam = fams{i};
        rows = S(strcmp(S.Family,fam),:);
        for j = 1:height(rows)
            fmt = '    %s & %s & %.3f & %.3f & %.3f & %.0f & %d \\\\\n';
            if j > 1, famLabel = ' '; else, famLabel = fam; end
            fprintf(fid, fmt, famLabel, rows.Controller{j}, rows.t_esc_s(j), rows.W_mech_J(j), ...
                           rows.theta_os_rad(j), rows.Sat_time_ms(j), rows.N_B(j));
        end
    end

    fprintf(fid, '    \\bottomrule\n  \\end{tabular}\n\\end{table}\n');
    fclose(fid);
end

function key = matKey(s)
% Safe struct field key for dynamic fieldnames
    if isstring(s) && isscalar(s), s = char(s); end
    key = matlab.lang.makeValidName(s);
end

function pts = ellipse_points(mu, C, n)
% 95% ellipse points around mu for 2x2 covariance C
    if nargin<3, n=200; end
    mu = mu(:)';
    C  = double(C);
    if any(~isfinite(C(:))) || ~isequal(size(C),[2 2])
        C = eye(2)*1e-8;
    end
    % 95% quantile of chi^2 with 2 dof
    k2 = 5.991;
    [V,D] = eig((C + C')/2);
    D = max(real(diag(D)), 1e-12);
    L = V * diag(sqrt(D*k2));
    th = linspace(0,2*pi,n);
    circ = [cos(th); sin(th)];
    E = (L * circ) + mu';
    pts = E.';
end

function p = spearman_perm_pval(a, b)
% Exact (or Monte Carlo) permutation p-value for Spearman rho.
% a,b: column vectors, same length m. For m<=7 we enumerate all permutations.
    a = a(:); b = b(:);
    m = numel(a);
    if m < 2 || any(~isfinite(a)) || any(~isfinite(b))
        p = NaN; return;
    end
    r_obs = corr(a, b, 'Type','Spearman', 'Rows','pairwise');

    if factorial(m) <= 5040  % exact up to m=7
        idxPerm = perms(1:m);
        r_all = zeros(size(idxPerm,1),1);
        for i = 1:size(idxPerm,1)
            r_all(i) = corr(a, b(idxPerm(i,:)), 'Type','Spearman','Rows','pairwise');
        end
        p = mean(abs(r_all) >= abs(r_obs));
        p = max(p, 1/size(idxPerm,1));  % avoid zero
    else
        K = 5000; r_all = zeros(K,1);
        for i = 1:K
            r_all(i) = corr(a, b(randperm(m)), 'Type','Spearman','Rows','pairwise');
        end
        p = max(mean(abs(r_all) >= abs(r_obs)), 1/K);
    end
end
