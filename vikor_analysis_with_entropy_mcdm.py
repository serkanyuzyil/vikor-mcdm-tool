import pandas as pd
import numpy as np
import os

def vikor_comprehensive_analysis(df_input, v_strategy=0.5):
    """
    VIKOR Method:
    1. Data Reading and Cleaning
    2. Weighting (Manual or Entropy)
    3. Calculation of Si, Ri, Qi
    4. Checking Conditions C1 and C2 (Step 6)
    """
    df = df_input.copy()
    
    # --- SECTION 1: DATA PREPARATION ---
    weight_index = None
    type_index = None
    
    for idx in df.index:
        idx_str = str(idx).lower().strip()
        if "weight" in idx_str:
            weight_index = idx
        elif "type" in idx_str:
            type_index = idx

    s_j_raw = df.loc[weight_index] if weight_index else None # raw subjective weight
    types_raw = df.loc[type_index] if type_index else None #loc location

    # Clean Matrix
    df_clean = df.drop(index=[x for x in [weight_index, type_index] if x is not None])
    decision_matrix = df_clean.apply(pd.to_numeric, errors='coerce')
    decision_matrix = decision_matrix.dropna(axis=1, how='all').dropna(axis=0, how='all')
    
    m, n = decision_matrix.shape

    # --- SECTION 2: PARAMETERS (Types and Weights) ---
    # Types
    if types_raw is not None:
        types = types_raw[decision_matrix.columns].astype(str).str.lower().str.strip()
    else:
        types = pd.Series(['max']*n, index=decision_matrix.columns)

    # Weights
    if s_j_raw is not None:
        manual_weights = pd.to_numeric(s_j_raw, errors='coerce').fillna(0)
        w_star = manual_weights / manual_weights.sum() if manual_weights.sum() != 0 else manual_weights
        mode_msg = "Manuel Ağırlıklar (Makale Modu)"
    else:
        # Entropy reserve
        r_temp = decision_matrix.divide(decision_matrix.sum(axis=0), axis=1)
        k_temp = 1.0 / np.log(m)
        Ej_temp = -k_temp * (r_temp * np.log(r_temp + 1e-15)).sum(axis=0)
        w_star = (1 - Ej_temp) / (1 - Ej_temp).sum()
        mode_msg = "Otomatik Entropi Ağırlıkları"

    # --- CHAPTER 3: VIKOR MATHEMATICS (Si, Ri, Qi)---
    f_max = decision_matrix.max()
    f_min = decision_matrix.min()
    
    vikor_norm = pd.DataFrame(index=decision_matrix.index, columns=decision_matrix.columns)
    
    for col in decision_matrix.columns:
        rng = f_max[col] - f_min[col]
        if rng == 0: rng = 1e-9
        
        if 'min' in str(types[col]): # Cost
            vikor_norm[col] = (decision_matrix[col] - f_min[col]) / rng
        else: # Benefit
            vikor_norm[col] = (f_max[col] - decision_matrix[col]) / rng

    weighted_vikor = vikor_norm * w_star
    Si = weighted_vikor.sum(axis=1)
    Ri = weighted_vikor.max(axis=1)

    # Qi Calculation
    S_best, S_worst = Si.min(), Si.max()
    R_best, R_worst = Ri.min(), Ri.max()
    
    term1 = (Si - S_best) / ((S_worst - S_best) if (S_worst - S_best) != 0 else 1e-9)
    term2 = (Ri - R_best) / ((R_worst - R_best) if (R_worst - R_best) != 0 else 1e-9)
    
    Qi = (v_strategy * term1) + ((1 - v_strategy) * term2)
    
    # Combine and Sort Results
    results = pd.DataFrame({'Si': Si, 'Ri': Ri, 'Qi': Qi})
    results['Rank'] = results['Qi'].rank(method='min', ascending=True).astype(int)
    results = results.sort_values(by='Qi', ascending=True)

    # --- SECTION 4: REFEREE REVIEW (C1 and C2 Conditions) ---
    # Best (a1) and Second Best (a2)
    a1_name = results.index[0]
    a2_name = results.index[1]
    
    Q_a1 = results.iloc[0]['Qi']
    Q_a2 = results.iloc[1]['Qi']
    
    # DQ Calculation (Threshold Value)
    DQ = 1 / (m - 1)
    
    # Reporting Text
    report = []
    report.append(f"--- VIKOR ANALİZ RAPORU ({mode_msg}) ---")
    report.append(f"Toplam Alternatif Sayısı (m): {m}")
    report.append(f"Eşik Değer (DQ = 1/(m-1)): {DQ:.4f}")
    report.append("-" * 30)
    report.append(f"1. Sıra (Kazanan): {a1_name} (Q = {Q_a1:.4f})")
    report.append(f"2. Sıra (Yedek):   {a2_name} (Q = {Q_a2:.4f})")
    report.append("-" * 30)

    # CONTROL 1: C1 (Acceptable Advantage)
    # Q(a2) - Q(a1) >= DQ
    diff = Q_a2 - Q_a1
    c1_status = diff >= DQ
    report.append(f"KOŞUL C1 (Avantaj): {'BAŞARILI' if c1_status else 'BAŞARISIZ'}")
    report.append(f" -> Fark ({diff:.4f}) >= DQ ({DQ:.4f}) ? {'Evet' if c1_status else 'Hayır'}")

    # CONTROL 2: C2 (Acceptable Stability)
    # a1 must also be the best (first) in either the Si or Ri ranking.
    # Note: You can have more than one 1st place in either Si or Ri, as long as you are among the best.
    best_S_val = results['Si'].min()
    best_R_val = results['Ri'].min()
    
    # Si and Ri values ​​of a1
    a1_Si = results.loc[a1_name, 'Si']
    a1_Ri = results.loc[a1_name, 'Ri']
    
    # Tolerant comparison (to avoid floating point errors)
    is_best_in_S = abs(a1_Si - best_S_val) < 1e-9
    is_best_in_R = abs(a1_Ri - best_R_val) < 1e-9
    
    c2_status = is_best_in_S or is_best_in_R
    
    report.append(f"CONDITION C2 (Stability)): {'SUCCESSFUL' if c2_status else 'UNSUCCESSFUL'}")
    report.append(f" -> {a1_name} Is Si ranked #1? {'Yes' if is_best_in_S else 'No'}")
    report.append(f" -> {a1_name} Is Ri ranked #1? {'Yes' if is_best_in_R else 'No'}")
    
    report.append("-" * 30)
    report.append("FINAL DECISION:")
    
    final_winners = []
    if c1_status and c2_status:
        report.append(f" >> THE ONLY AND ABSOLUTE WINNER: {a1_name}")
        final_winners.append(a1_name)
    else:
        # Compromise Solution Set
        if not c1_status:
            report.append(" >>C1 Not Provided! 'Advantage' Insufficient.")
            report.append(" >> A set of winning participants (Compromise Solutions) is being created:")
            #Everyone whose Q(ai) - Q(a1) < DQ is a winner.
            for idx, row in results.iterrows():
                if (row['Qi'] - Q_a1) < DQ:
                    report.append(f"    * {idx} (Fark {row['Qi'] - Q_a1:.4f} < DQ)")
                    final_winners.append(idx)
        elif not c2_status:
             report.append(" >> C2 Failed! 'Stability' insufficient.")
             report.append(f" >> Winners: {a1_name} ve {a2_name}")
             final_winners = [a1_name, a2_name]

    return results, w_star, decision_matrix, "\n".join(report)

# ==========================================
# WORK AREA
# ==========================================
calisma_klasoru = os.path.dirname(os.path.abspath(__file__))
dosya_adi = 'flywheel_four_criteria.xlsx' 
tam_yol = os.path.join(calisma_klasoru, dosya_adi)

print(f"In progress: {tam_yol}")

try:
    df_raw = pd.read_excel(tam_yol, index_col=0).dropna(how='all')
    
    # Calculate
    sonuclar, agirliklar, veri, metin_raporu = vikor_comprehensive_analysis(df_raw, v_strategy=0.5)
    
    print("\n" + "="*50)
    print(metin_raporu) # Print the detailed report to the console.
    print("="*50)
    
    # Save
    dosya_ismi = "VIKOR_Full_Analysis_Report.xlsx"
    cikti_yolu = os.path.join(calisma_klasoru, dosya_ismi)
    
    with pd.ExcelWriter(cikti_yolu) as writer:
        sonuclar.to_excel(writer, sheet_name='Siralama_ve_Skorlar')
        agirliklar.to_excel(writer, sheet_name='Agirliklar')
        
        # Let's write the report as text in a separate Excel sheet.
        # We convert and write it to Pandas DataFrame.
        rapor_df = pd.DataFrame([row for row in metin_raporu.split('\n')], columns=["Analysis_Report"])
        rapor_df.to_excel(writer, sheet_name='Referee_Report', index=False)
        
    print(f"\nThe file has been saved: {dosya_ismi}")

except Exception as e:
    print(f"HATA: {e}")