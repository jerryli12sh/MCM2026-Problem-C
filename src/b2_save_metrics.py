import numpy as np
import pandas as pd

# ---------- helpers ----------

def get_Tmax_by_season(panel: pd.DataFrame) -> dict:
    return panel.groupby('season')['week'].max().to_dict()


def _infer_alive_flags(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    if 'alive' not in df.columns:
        df['alive'] = True
    if 'is_final_week' in df.columns:
        df['is_finale'] = df['is_final_week'].astype(bool)
    elif 'is_finale' in df.columns:
        df['is_finale'] = df['is_finale'].astype(bool)
    elif 'is_final_week_end' in df.columns:
        df['is_finale'] = df['is_final_week_end'].astype(bool)
    else:
        df['is_finale'] = False
    if 'elim_this_week_end' in df.columns:
        g = df.groupby(['season', 'week'])['elim_this_week_end'].sum().reset_index(name='elim_cnt')
        df = df.merge(g, on=['season', 'week'], how='left')
        df['is_single_elim_week'] = df['elim_cnt'] == 1
    elif 'is_single_elim_week' in df.columns:
        df['is_single_elim_week'] = df['is_single_elim_week'].astype(bool)
    else:
        df['is_single_elim_week'] = True
    return df


def eligible_week_selector(df_week: pd.DataFrame) -> bool:
    if 'is_finale' in df_week.columns and df_week['is_finale'].iloc[0]:
        return False
    if 'is_single_elim_week' in df_week.columns and not df_week['is_single_elim_week'].iloc[0]:
        return False
    if 'alive' in df_week.columns:
        return df_week['alive'].sum() > 2
    return True


def _get_judge_score_cols(df: pd.DataFrame):
    for col in ['total_judge_score', 'judge_total', 'j_metric', 'judge_score', 'score_judge']:
        if col in df.columns:
            return col
    return None


def _rank_desc(x):
    return pd.Series(x).rank(ascending=False, method='average').to_numpy()


def _vec_from_col(df: pd.DataFrame, names, col: str):
    m = df.set_index('celebrity_name')[col]
    return m.reindex(names).to_numpy(float)


def _align_df_to_names(df: pd.DataFrame, names):
    order = pd.Index(names)
    d = df.copy()
    d['celebrity_name'] = pd.Categorical(d['celebrity_name'], categories=order, ordered=True)
    d = d.sort_values('celebrity_name')
    return d


def get_week_draws(posterior_draws, season, week, names):
    key = (int(season), int(week))
    if isinstance(posterior_draws, dict) and key in posterior_draws:
        val = posterior_draws[key]
        if isinstance(val, dict) and 'p_draws' in val and 'names' in val:
            p = val['p_draws']
            draw_names = list(val['names'])
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            p, draw_names = val
        else:
            p, draw_names = val, names
        if list(draw_names) != list(names):
            idx = {n: i for i, n in enumerate(draw_names)}
            cols = [idx[n] for n in names if n in idx]
            p = p[:, cols]
            names = [n for n in names if n in idx]
        return p, names

    if isinstance(posterior_draws, pd.DataFrame):
        required = {'season', 'week', 'celebrity_name', 'b', 'p_draw'}
        if required.issubset(posterior_draws.columns):
            g = posterior_draws[(posterior_draws['season'] == season) & (posterior_draws['week'] == week)]
            if g.empty:
                return None, None
            order = pd.Index(names)
            g = g.copy()
            g['celebrity_name'] = pd.Categorical(g['celebrity_name'], categories=order, ordered=True)
            g = g.sort_values(['b', 'celebrity_name'])
            p = g.pivot_table(index='b', columns='celebrity_name', values='p_draw')
            p = p.reindex(columns=order)
            return p.to_numpy(), list(p.columns)

    return None, None


def compute_risk_and_bottom2(p_draw, names, judge_pct, judge_rank, baseline_mode, wJ=0.5, wF=0.5):
    names = np.asarray(names)
    p_draw = np.asarray(p_draw)
    judge_pct = np.asarray(judge_pct)
    judge_rank = np.asarray(judge_rank)

    n = len(names)
    if n == 0:
        return None, None, None, None
    if len(p_draw) != n or len(judge_pct) != n or len(judge_rank) != n:
        return None, None, None, None

    if baseline_mode == "pct":
        risk = wJ * (1.0 - judge_pct) + wF * (1.0 - p_draw)
    else:
        fan_rank = _rank_desc(p_draw)
        risk = wJ * judge_rank + wF * fan_rank

    name_key = np.argsort(names)
    order = np.lexsort((name_key, p_draw, judge_pct, -risk))
    if order.size == 0:
        return None, None, None, None

    bottom2_idx = order[:2]
    bottom2 = list(names[bottom2_idx])
    elim_base = names[order[0]]

    if baseline_mode == "pct":
        judge_signal = judge_pct
    else:
        judge_signal = -judge_rank

    b2_j = judge_signal[bottom2_idx]
    b2_p = p_draw[bottom2_idx]
    b2_names = names[bottom2_idx]
    elim_idx = bottom2_idx[np.lexsort((np.argsort(b2_names), b2_p, b2_j))[0]]
    elim_save = names[elim_idx]

    return risk, bottom2, elim_base, elim_save


def _judge_vectors(g: pd.DataFrame, names, baseline_mode: str):
    if baseline_mode == 'pct':
        if 'judge_pct' in g.columns:
            judge_pct = _vec_from_col(g, names, 'judge_pct')
        elif 'judge_share' in g.columns:
            judge_pct = _vec_from_col(g, names, 'judge_share')
        else:
            score_col = _get_judge_score_cols(g)
            if score_col is None:
                raise KeyError('No judge score column found. Expected one of: total_judge_score, judge_total, j_metric, judge_score, score_judge')
            v = _vec_from_col(g, names, score_col)
            judge_pct = v / v.sum() if v.sum() != 0 else np.zeros_like(v)
        judge_rank = _rank_desc(judge_pct)
    else:
        if 'judge_rank' in g.columns:
            judge_rank = _vec_from_col(g, names, 'judge_rank')
        else:
            score_col = _get_judge_score_cols(g)
            if score_col is None:
                raise KeyError('No judge score column found. Expected one of: total_judge_score, judge_total, j_metric, judge_score, score_judge')
            judge_rank = _rank_desc(_vec_from_col(g, names, score_col))
        judge_pct = 1.0 - (judge_rank - 1.0) / max(len(judge_rank) - 1, 1)
    return judge_pct, judge_rank


# ---------- main ----------

def compute_b2_save_metrics(
    selected_pairs,
    panel: pd.DataFrame,
    posterior_draws,
    baseline_modes=("rank", "pct"),
    wJ=0.5,
    wF=0.5,
    contestant_types: pd.DataFrame = None,
    bootstrap=False,
    n_boot=1000,
):
    df = _infer_alive_flags(panel)
    Tmax_by_season = get_Tmax_by_season(df)

    rows = []

    for baseline_mode in baseline_modes:
        for season, name in selected_pairs:
            season = int(season)
            Tmax = Tmax_by_season.get(season)
            if Tmax is None:
                continue

            weeks = sorted([wk for wk in df[df['season'] == season]['week'].unique() if wk < Tmax])
            weeks = [wk for wk in weeks if eligible_week_selector(df[(df['season'] == season) & (df['week'] == wk)])]

            p_b2_hits = 0
            p_rev_given_b2_hits = 0
            p_rev_hits = 0
            denom_b2 = 0
            T_i = 0
            B = None

            for wk in weeks:
                g = df[(df['season'] == season) & (df['week'] == wk) & (df['alive'] == True)].copy()
                if g.empty:
                    continue
                names = g['celebrity_name'].tolist()
                if name not in names:
                    continue

                p_draws, aligned_names = get_week_draws(posterior_draws, season, wk, names)
                if p_draws is None:
                    continue

                names = aligned_names
                g = g[g['celebrity_name'].isin(names)].copy()
                g = _align_df_to_names(g, names)
                if name not in names:
                    continue

                judge_pct, judge_rank = _judge_vectors(g, names, baseline_mode)

                B = p_draws.shape[0]
                for b in range(B):
                    p_b = p_draws[b]
                    _, bottom2, elim_base, elim_save = compute_risk_and_bottom2(
                        p_b, names, judge_pct, judge_rank, baseline_mode, wJ=wJ, wF=wF
                    )
                    if elim_base is None:
                        continue
                    if name in bottom2:
                        p_b2_hits += 1
                        denom_b2 += 1
                        if elim_base != elim_save:
                            p_rev_given_b2_hits += 1
                    if elim_base != elim_save:
                        p_rev_hits += 1
                T_i += 1

            if T_i == 0 or B is None:
                continue

            p_b2 = p_b2_hits / (T_i * B)
            p_rev = p_rev_hits / (T_i * B)
            p_rev_given_b2 = (p_rev_given_b2_hits / denom_b2) if denom_b2 > 0 else np.nan

            # season-level per draw simulation
            weeks_sim = weeks
            if not weeks_sim:
                continue

            alive0 = set(df[(df['season'] == season) & (df['week'] == weeks_sim[0]) & (df['alive'] == True)]['celebrity_name'].tolist())
            p0, names0 = get_week_draws(posterior_draws, season, weeks_sim[0], list(alive0))
            if p0 is None:
                continue
            B = p0.shape[0]

            t_base_draws = []
            t_save_draws = []

            for b in range(B):
                alive_base = set(alive0)
                alive_save = set(alive0)
                t_base = Tmax
                t_save = Tmax

                for wk in weeks_sim:
                    g_base = df[(df['season'] == season) & (df['week'] == wk) & (df['celebrity_name'].isin(alive_base))].copy()
                    if len(g_base) >= 3:
                        names = g_base['celebrity_name'].tolist()
                        p_draws, aligned_names = get_week_draws(posterior_draws, season, wk, names)
                        if p_draws is not None:
                            names = aligned_names
                            g_base = g_base[g_base['celebrity_name'].isin(names)].copy()
                            g_base = _align_df_to_names(g_base, names)
                            p_b = p_draws[b]
                            judge_pct, judge_rank = _judge_vectors(g_base, names, baseline_mode)
                            _, _, elim_base, _ = compute_risk_and_bottom2(
                                p_b, names, judge_pct, judge_rank, baseline_mode, wJ=wJ, wF=wF
                            )
                            if elim_base is not None:
                                if name in alive_base and elim_base == name and t_base == Tmax:
                                    t_base = wk
                                alive_base.discard(elim_base)

                    g_save = df[(df['season'] == season) & (df['week'] == wk) & (df['celebrity_name'].isin(alive_save))].copy()
                    if len(g_save) >= 3:
                        names = g_save['celebrity_name'].tolist()
                        p_draws, aligned_names = get_week_draws(posterior_draws, season, wk, names)
                        if p_draws is not None:
                            names = aligned_names
                            g_save = g_save[g_save['celebrity_name'].isin(names)].copy()
                            g_save = _align_df_to_names(g_save, names)
                            p_b = p_draws[b]
                            judge_pct, judge_rank = _judge_vectors(g_save, names, baseline_mode)
                            _, _, _, elim_save = compute_risk_and_bottom2(
                                p_b, names, judge_pct, judge_rank, baseline_mode, wJ=wJ, wF=wF
                            )
                            if elim_save is not None:
                                if name in alive_save and elim_save == name and t_save == Tmax:
                                    t_save = wk
                                alive_save.discard(elim_save)

                t_base_draws.append(t_base)
                t_save_draws.append(t_save)

            t_base_draws = np.asarray(t_base_draws)
            t_save_draws = np.asarray(t_save_draws)
            dE_T = float(np.mean(t_save_draws) - np.mean(t_base_draws))
            dP_finals = float(np.mean(t_save_draws == Tmax) - np.mean(t_base_draws == Tmax))

            rows.append({
                'season': season,
                'celebrity_name': name,
                'baseline_mode': baseline_mode,
                'p_b2': p_b2,
                'p_rev_given_b2': p_rev_given_b2,
                'p_rev': p_rev,
                'dE_T': dE_T,
                'dP_finals': dP_finals,
                'n_weeks': T_i,
                'B': int(B),
                'denom_b2': int(denom_b2),
                'Tmax': int(Tmax),
                'unit_type': 'individual',
                'unit_id': name,
            })

    out = pd.DataFrame(rows)

    if contestant_types is not None:
        type_df = contestant_types.copy()
        type_df['season'] = type_df['season'].astype(int)
        merged = out.merge(type_df, on=['season', 'celebrity_name'], how='left')

        group_rows = []
        metrics = ['p_b2', 'p_rev_given_b2', 'p_rev', 'dE_T', 'dP_finals']
        for baseline_mode in baseline_modes:
            g0 = merged[merged['baseline_mode'] == baseline_mode]
            for t, g in g0.groupby('contestant_type'):
                row = {
                    'season': np.nan,
                    'celebrity_name': np.nan,
                    'baseline_mode': baseline_mode,
                    'unit_type': 'group',
                    'unit_id': t,
                    'n_units': int(len(g)),
                }
                for m in metrics:
                    row[f'{m}_mean'] = float(np.nanmean(g[m])) if len(g) else np.nan
                    row[f'{m}_median'] = float(np.nanmedian(g[m])) if len(g) else np.nan
                if bootstrap and len(g) > 1:
                    rng = np.random.default_rng(42)
                    for m in metrics:
                        boots = []
                        vals = g[m].to_numpy()
                        for _ in range(n_boot):
                            samp = rng.choice(vals, size=len(vals), replace=True)
                            boots.append(np.nanmean(samp))
                        lo, hi = np.nanquantile(boots, [0.025, 0.975])
                        row[f'{m}_mean_lo'] = float(lo)
                        row[f'{m}_mean_hi'] = float(hi)
                group_rows.append(row)

        out = pd.concat([out, pd.DataFrame(group_rows)], ignore_index=True)

    return out
