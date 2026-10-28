#!/usr/bin/env python3
"""HAI 22.04: analise EXPLORATORIA de detectores locais individuais.

Nao executa RNN; nao altera diagnostico normal-only ou avaliacao principal.
Dois regimes:
 (1) contribuicao dos grupos sob limiar congelado da fusao MAX;
 (2) quatro detectores individuais, cada um com P99 da validacao NORMAL.
Valida que Global e MAX reproduzem EXATAMENTE os resultados principais.
Cria pasta nova exclusivamente apos os calculos; nunca sobrescreve.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE / 'resultado/diagnostico_mahalanobis_normal_only'
TREINO = ROOT / 'residuos_treino_assinados.npy'
VALIDACAO = BASE / 'resultado/rnn_hai/secaoMONOLITICO/residuos_monolitico/residuos_validacao_monolitico.csv'
TESTE = BASE / 'resultado/rnn_hai/secaoMONOLITICO/residuos_monolitico/residuos_teste_monolitico.csv'
CABECALHO = BASE / 'dataset_trabalho/hai-22.04/normalizados_oficiais/treino/train1_treino_normalizado.csv'
DETALHE = ROOT / 'diagnostico/diagnostico_por_grupo.csv'
RESUMO = ROOT / 'diagnostico/diagnostico_resumo.csv'
PRINCIPAL_METRICAS = ROOT / 'avaliacao_ataques/metricas_global_local.csv'
PRINCIPAL_EVENTOS = ROOT / 'avaliacao_ataques/comparacao_58_eventos.csv'
DESTINO = ROOT / 'analise_exploratoria_por_grupo'
HASHES = {
    DETALHE: '5d9713a1d89065585e66dc02dfb4a42646be646779b615c55430a29951430901',
    RESUMO: 'ac87726c478f6273eb4d5971828b41270cbfda0b61637d767801da5d04bdaf1a',
}
TAMANHOS = {'P1': 44, 'P2': 24, 'P3': 7, 'P4': 11}
FPR_ALVO = 0.01
N_TREINO = 805692
N_VAL = 197646
N_TESTE = 360964
N_ATAQUES = 12030
N_EVENTOS = 58
BLOCO_TESTE = 10000


def exigir(condicao, mensagem):
    if not condicao:
        raise RuntimeError(mensagem)


def sha256(arquivo):
    h = hashlib.sha256()
    with arquivo.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def obter_features():
    colunas = pd.read_csv(CABECALHO, nrows=0).columns.str.strip().tolist()
    features = [c for c in colunas if c not in ('timestamp', 'Attack', 'Tipo')]
    exigir(len(features) == 86 and len(set(features)) == 86, 'Features invalidas.')
    indices = {}
    for g, tamanho in TAMANHOS.items():
        idx = np.array([i for i, f in enumerate(features) if f.startswith(g + '_')], dtype=np.int64)
        exigir(len(idx) == tamanho, f'Grupo {g} invalido.')
        indices[g] = idx
    exigir(np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(86)),
           'Particao incompleta ou sobreposta.')
    return features, indices


def metricas(cont):
    tn, fp, fn, tp = (int(x) for x in cont.ravel())
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    fpr = fp/(fp+tn) if fp+tn else 0.
    denom = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5
    mcc = (tp*tn-fp*fn)/denom if denom else 0.
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, Precision=precision,
                Recall=recall, F1=f1, MCC=mcc, FPR=fpr)


def main():
    exigir(not DESTINO.exists() and not DESTINO.is_symlink(),
           f'Destino ja existe; nada sera sobrescrito: {DESTINO}')
    print('=== INTEGRIDADE E PROTOCOLO PRINCIPAL ===', flush=True)
    for arquivo, esperado in HASHES.items():
        exigir(sha256(arquivo) == esperado, f'Hash divergente: {arquivo}')
        print('SHA-256 OK:', arquivo.name, flush=True)
    resumo = pd.read_csv(RESUMO)
    exigir(len(resumo) == 1, 'Resumo nao possui uma linha.')
    r = resumo.iloc[0]
    exigir(str(r['Dataset']) == 'HAI_22.04' and str(r['Precisao_Estatistica']) == 'float64'
           and int(r['N_Grupos']) == 4 and abs(float(r['FPR_Alvo']) - FPR_ALVO) < 1e-15,
           'Protocolo do diagnostico divergente.')
    tau_g, tau_max = float(r['Tau_Global']), float(r['Tau_Local'])
    exigir(np.isclose(tau_g, 199.4393418878611, atol=1e-10, rtol=0)
           and np.isclose(tau_max, 135.92393580193848, atol=1e-10, rtol=0),
           'Limiares principais divergentes.')
    exigir(int(r['Alarmes_Val_Global']) == 1977 and int(r['Alarmes_Val_Local']) == 1977,
           'Orcamento de validacao divergente.')
    principais_m = pd.read_csv(PRINCIPAL_METRICAS).set_index('Detector')
    principais_e = pd.read_csv(PRINCIPAL_EVENTOS).set_index('Evento')
    exigir(set(principais_m.index) == {'Mahalanobis_Global', 'Mahalanobis_Local_Max'},
           'Metricas principais inesperadas.')
    exigir(len(principais_e) == N_EVENTOS and principais_e.index.is_unique,
           'Eventos principais inesperados.')

    features, indices = obter_features()
    colunas_err = ['err_' + f for f in features]
    X_train = np.load(TREINO, allow_pickle=False)
    exigir(X_train.shape == (N_TREINO, 86) and X_train.dtype == np.float32
           and np.isfinite(X_train).all(), 'Treino invalido.')
    X_train = X_train.astype(np.float64)
    val = pd.read_csv(VALIDACAO, usecols=colunas_err + ['Attack'],
                      dtype={c: np.float32 for c in colunas_err})
    exigir(len(val) == N_VAL and list(val[colunas_err].columns) == colunas_err,
           'Validacao invalida.')
    exigir(pd.to_numeric(val['Attack'], errors='raise').eq(0).all(),
           'Validacao contem ataques.')
    X_val = val[colunas_err].to_numpy(dtype=np.float64)
    exigir(np.isfinite(X_val).all(), 'Residuos nao finitos na validacao.')
    del val

    print('Ajustando global e quatro locais (LedoitWolf float64)...', flush=True)
    global_model = LedoitWolf(store_precision=True, assume_centered=False).fit(X_train)
    modelos = {g: LedoitWolf(store_precision=True, assume_centered=False).fit(X_train[:, idx])
               for g, idx in indices.items()}
    del X_train

    # Calibracao independente dos quatro grupos: NAO usa dados de teste.
    scores_val = {g: modelos[g].mahalanobis(X_val[:, idx]) for g, idx in indices.items()}
    tau_ind = {g: float(np.quantile(score, 1.0 - FPR_ALVO))
               for g, score in scores_val.items()}
    val_alarmes_ind = {g: int(np.count_nonzero(scores_val[g] > tau_ind[g])) for g in TAMANHOS}
    sl_val = np.max(np.column_stack([scores_val[g] for g in TAMANHOS]), axis=1)
    sg_val = global_model.mahalanobis(X_val)
    exigir(np.isclose(float(np.quantile(sl_val, .99)), tau_max, rtol=0, atol=1e-8)
           and np.isclose(float(np.quantile(sg_val, .99)), tau_g, rtol=0, atol=1e-8),
           'Nao foi possivel reproduzir os limiares congelados.')
    exigir(int(np.count_nonzero(sl_val > tau_max)) == 1977
           and int(np.count_nonzero(sg_val > tau_g)) == 1977,
           'Alarme de validacao principal divergente.')
    print('Validacao reproduzida: global/MAX = 1977 alarmes cada.', flush=True)
    for g in TAMANHOS:
        print(f'  {g}: limiar isolado P99={tau_ind[g]:.15f}; '
              f'alarmes val={val_alarmes_ind[g]}/{N_VAL}', flush=True)
    del X_val, scores_val, sg_val, sl_val

    detectores = ['Global', 'Local_MAX'] + [f'{g}_na_MAX' for g in TAMANHOS] + [f'{g}_isolado' for g in TAMANHOS]
    cont = {d: np.zeros((2, 2), dtype=np.int64) for d in detectores}
    eventos = {}  # Tipo -> colunas de alarmes/deteccao por detector
    total = ataques = intervalos = 0
    anterior = None
    print('=== AVALIACAO EXPLORATORIA DO TESTE ===', flush=True)
    leitor = pd.read_csv(TESTE,
                         usecols=['arquivo_origem', 'Tipo', 'Attack'] + colunas_err,
                         dtype={**{c: np.float32 for c in colunas_err},
                                'Tipo': 'string', 'arquivo_origem': 'string'},
                         chunksize=BLOCO_TESTE)
    for numero_bloco, df in enumerate(leitor, 1):
        exigir(list(df[colunas_err].columns) == colunas_err, 'Ordem de features divergente.')
        X = df[colunas_err].to_numpy(dtype=np.float64)
        exigir(np.isfinite(X).all(), 'Residuos nao finitos no teste.')
        a = pd.to_numeric(df['Attack'], errors='raise')
        exigir(a.notna().all(), 'Attack ausente.')
        y = a.ne(0).to_numpy(dtype=bool)
        tipos = df['Tipo'].astype('string')
        origens = df['arquivo_origem'].astype('string')
        exigir(tipos[y].notna().all() and tipos[y].str.strip().ne('').all(),
               'Tipo de ataque ausente.')
        exigir(origens.notna().all(), 'arquivo_origem ausente.')

        sg = global_model.mahalanobis(X)
        por_grupo = {g: modelos[g].mahalanobis(X[:, idx]) for g, idx in indices.items()}
        sl = np.max(np.column_stack([por_grupo[g] for g in TAMANHOS]), axis=1)
        preds = {'Global': sg > tau_g, 'Local_MAX': sl > tau_max}
        for g in TAMANHOS:
            preds[f'{g}_na_MAX'] = por_grupo[g] > tau_max
            preds[f'{g}_isolado'] = por_grupo[g] > tau_ind[g]
        exigir(np.array_equal(np.logical_or.reduce([preds[f'{g}_na_MAX'] for g in TAMANHOS]),
                                  preds['Local_MAX']), 'Fusao MAX divergente.')
        for d, p in preds.items():
            cont[d][0, 0] += np.count_nonzero(~y & ~p)
            cont[d][0, 1] += np.count_nonzero(~y & p)
            cont[d][1, 0] += np.count_nonzero(y & ~p)
            cont[d][1, 1] += np.count_nonzero(y & p)

        # Mesma contagem dos eventos do avaliador principal, incluindo fronteiras de blocos.
        for i in range(len(df)):
            estado = (str(origens.iloc[i]), bool(y[i]), str(tipos.iloc[i]))
            if estado[1]:
                nome = estado[2]
                info = eventos.setdefault(nome, {'Evento': nome, 'Amostras_Ataque': 0,
                    **{f'Alarmes_{d}': 0 for d in detectores}})
                info['Amostras_Ataque'] += 1
                for d in detectores:
                    info[f'Alarmes_{d}'] += int(preds[d][i])
                if anterior != estado:
                    intervalos += 1
            anterior = estado
        total += len(df)
        ataques += int(np.count_nonzero(y))
        if numero_bloco % 10 == 0:
            print(f'  {total} linhas avaliadas...', flush=True)

    exigir(total == N_TESTE and ataques == N_ATAQUES, 'Quantidade de amostras inesperada.')
    exigir(len(eventos) == N_EVENTOS and intervalos == N_EVENTOS,
           'Quantidade de eventos inesperada.')
    df_e = pd.DataFrame(eventos.values()).sort_values('Evento').reset_index(drop=True)
    for d in detectores:
        df_e[f'Detectado_{d}'] = df_e[f'Alarmes_{d}'] > 0
    exigir(df_e['Amostras_Ataque'].sum() == N_ATAQUES, 'Soma de amostras de ataque invalida.')

    # Regressao: comparacao EXATA dos resultados ja publicados.
    exigir(set(df_e['Evento']) == set(principais_e.index), 'Eventos nao coincidem com avaliacao principal.')
    comparados = df_e.set_index('Evento').loc[principais_e.index]
    for d, principal in [('Global', 'Mahalanobis_Global'), ('Local_MAX', 'Mahalanobis_Local_Max')]:
        m = metricas(cont[d])
        for k in ('TP', 'FP', 'TN', 'FN'):
            exigir(m[k] == int(principais_m.loc[principal, k]),
                   f'Metrica principal divergente: {d} {k}')
        for k in ('Precision', 'Recall', 'F1', 'MCC', 'FPR'):
            exigir(np.isclose(m[k], float(principais_m.loc[principal, k]), rtol=0, atol=1e-12),
                   f'Metrica principal divergente: {d} {k}')
        for nome in ('Alarmes', 'Detectado'):
            a = comparados[f'{nome}_{d}'].to_numpy()
            b = principais_e[f'{nome}_{"Global" if d == "Global" else "Local"}'].to_numpy()
            exigir(np.array_equal(a, b), f'Eventos principais divergentes: {d} {nome}')
    exigir(np.array_equal(comparados['Amostras_Ataque'].to_numpy(),
                                  principais_e['Amostras_Ataque'].to_numpy()),
           'Tamanhos dos eventos principais divergentes.')
    print('Regressao OK: Global e MAX reproduzem metricas e 58 eventos principais.', flush=True)

    linhas = []
    for d in detectores:
        dados = metricas(cont[d])
        dados.update(Detector=d, Eventos_Detectados=int(df_e[f'Detectado_{d}'].sum()),
                     Total_Eventos=N_EVENTOS,
                     Tau_Usado=tau_g if d == 'Global' else tau_max if d == 'Local_MAX' or d.endswith('_na_MAX')
                     else tau_ind[d.split('_')[0]],
                     Regime='Principal' if d in ('Global', 'Local_MAX') else
                     'Contribuicao_na_fusao_MAX' if d.endswith('_na_MAX') else 'Isolado_P99_exploratorio',
                     Alarmes_Val=1977 if d in ('Global', 'Local_MAX') else
                     None if d.endswith('_na_MAX') else val_alarmes_ind[d.split('_')[0]])
        linhas.append(dados)
    df_m = pd.DataFrame(linhas)
    # Quantidade real na validacao sob limiar MAX para cada grupo (explicita e informativa).
    # Recalcular apenas escores de validacao seria desnecessario para o experimento;
    # nao atribuir '0' como se fosse medicao. Usar NA.
    df_m.loc[df_m['Regime'] == 'Contribuicao_na_fusao_MAX', 'Alarmes_Val'] = np.nan

    resumo_intersecoes = []
    g = df_e['Detectado_Global'].to_numpy(dtype=bool)
    for d in detectores[2:]:
        q = df_e[f'Detectado_{d}'].to_numpy(dtype=bool)
        resumo_intersecoes.append({'Detector': d, 'Ambos': int(np.sum(g & q)),
            'Somente_Global': int(np.sum(g & ~q)),
            'Somente_Detector': int(np.sum(~g & q)),
            'Nenhum': int(np.sum(~g & ~q)),
            'Eventos_Exclusivos_Detector': ', '.join(df_e.loc[~g & q, 'Evento'])})
    df_inter = pd.DataFrame(resumo_intersecoes)

    protocolo = {'dataset': 'HAI_22.04', 'carater': 'analise exploratoria posterior a comparacao principal',
                'treino': 'residuos normais assinados da mesma RNN, LedoitWolf float64',
                'calibracao': 'normal only, quantil P99; limiares individuais diferentes do MAX',
                'tau_global_congelado': tau_g, 'tau_max_congelado': tau_max,
                'tau_individuais': tau_ind, 'alarmes_validacao_individuais': val_alarmes_ind,
                'regra': 'score > limiar', 'fusao': 'MAX',
                'contribuicao_na_MAX': 'escore de cada grupo > limiar da fusao; NAO e detector individual calibrado',
                'eventos': '58 valores Tipo, um intervalo cada',
                'integridade_diagnostico_sha256': {p.name: h for p, h in HASHES.items()},
                'checagem_regressao': 'matriz de confusao e 58 eventos Global/MAX identicos a avaliacao principal',
                'amostras_teste': total, 'amostras_ataque': ataques,
                'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__}
    # Pasta e arquivos novos exclusivamente, A POS calculos e checagens.
    DESTINO.mkdir(parents=False, exist_ok=False)
    with (DESTINO / 'metricas_por_detector.csv').open('x', encoding='utf-8') as f:
        df_m.to_csv(f, index=False)
    with (DESTINO / 'eventos_por_detector.csv').open('x', encoding='utf-8') as f:
        df_e.to_csv(f, index=False)
    with (DESTINO / 'intersecoes_com_global.csv').open('x', encoding='utf-8') as f:
        df_inter.to_csv(f, index=False)
    with (DESTINO / 'protocolo_exploratorio.json').open('x', encoding='utf-8') as f:
        json.dump(protocolo, f, indent=2, ensure_ascii=False)
    print('\n=== METRICAS POR DETECTOR ===')
    print(df_m[['Detector', 'Regime', 'TP', 'FP', 'Recall', 'F1', 'FPR', 'Eventos_Detectados']].to_string(index=False))
    print('\n=== ATAQUES DETECTADOS APENAS PELO DETECTOR COMPARADO AO GLOBAL ===')
    print(df_inter[['Detector', 'Somente_Detector', 'Eventos_Exclusivos_Detector']].to_string(index=False))
    print('\nArquivos novos exclusivamente em:', DESTINO)


if __name__ == '__main__':
    main()
