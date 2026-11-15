"""Patch/audit the remaining comparison and grid tables from current results.

Run analyze_remaining_materials first. Uses only current CSVs and the replay's
individual grid observations; no historical results or inferred missing data.
"""
import argparse
import csv
from decimal import Decimal
import difflib
from pathlib import Path

from export_monte_carlo_tables import (
    DATASETS, METHODS, RATIOS, SOURCE, GRID_SOURCE, load_csvs, reference_directory,
)

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'article/manuscrito/tables'
REPLAY = reference_directory('reviewer4')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def grid_data():
    rows = read_csv(REPLAY / 'grid_observations.csv')
    assert len(rows) == 1500
    lookup = {}
    for dataset, _ in DATASETS:
        for ratio in RATIOS:
            group = [r for r in rows if r['dataset'] == dataset and Decimal(r['ratio']) == Decimal(ratio)/100]
            assert len(group) == 50
            assert [int(r['seed']) for r in group] == list(range(1+50*(RATIOS.index(ratio)),51+50*RATIOS.index(ratio)))
            for r in group:
                assert int(r['cells']) == int(r['nx']) * int(r['ny'])
                assert int(r['clipped_lower']) == 0
                assert int(r['at_lower']) == int(int(r['L_min']) in [int(r['nx']),int(r['ny'])])
                assert int(r['at_upper']) == int(int(r['L_max']) in [int(r['nx']),int(r['ny'])])
                assert int(r['clipped_upper']) == int(max(int(r['requested_nx']),int(r['requested_ny'])) > int(r['L_max']))
            lookup[dataset, ratio] = group
    return lookup


def occupancy():
    groups = [
        (['franke','gauss_mixture'],10,'Franke / mistura gaussiana'),
        (['peaks'],10,'Peaks'), (['sinc'],10,'Sinc'),
        ([d for d,_ in DATASETS[:4]],30,'Quatro funções sintéticas'),
        ([d for d,_ in DATASETS[:4]],50,'Quatro funções sintéticas'),
        *[([d for d,_ in DATASETS[4:]],ratio,'NOAA (janeiro--junho)') for ratio in RATIOS],
    ]
    path = TABLES / 'grid_occupancy.tex'
    old = path.read_text(encoding='utf-8').splitlines()
    start = old.index(r'    \midrule')
    end = old.index(r'    \bottomrule')
    lines = ['% Source: current Random CSV grid columns; independently replayed from training coordinates.'] + old[1:start+1]
    for datasets, ratio, label in groups:
        values = set()
        for dataset in datasets:
            records = read_csv(SOURCE / f'errors_{dataset}_ratio{ratio}_random.csv')
            ts = [r for r in records if r['Row'] in METHODS[4:]]
            for r in ts:
                assert r['N_grid'] == '50'
                values.add(tuple(Decimal(r[key]) for key in ['MeanGridX1','MeanGridX2','MeanGridCells','MeanEmptyCellsPct']))
        assert len(values) == 1, (label,ratio,values)
        nx,ny,cells,empty = values.pop()
        lines.append(f'    {label} & ${ratio}\\%$ & {4*ratio} & ${nx:.2f}\\times{ny:.2f}$ & {cells:.2f} & {empty:.3f} ' + r'\\')
    lines += old[end:]
    return '\n'.join(lines)+'\n'


def limits_table(lookup):
    old = (TABLES/'reviewer1_grid_limits.tex').read_text(encoding='utf-8').splitlines()
    first = old.index(r'\midrule')
    bottom = old.index(r'\bottomrule')
    lines = [f'% Source: {(REPLAY / "grid_observations.csv").relative_to(ROOT).as_posix()}.'] + old[1:first+1]
    for subset,label in [(DATASETS[:4],'Quatro sintéticos'),(DATASETS[4:],'Seis campos NOAA')]:
        if subset == DATASETS[4:]: lines.append(r'\midrule')
        for ratio in RATIOS:
            group = [r for d,_ in subset for r in lookup[d,ratio]]
            maximum = {r['L_max'] for r in group}
            assert len(maximum) == 1
            counts = [sum(int(r[key]) for r in group) for key in ['at_lower','at_upper','clipped_upper']]
            counts.append(sum(int(r['empty_rows'])>0 or int(r['empty_columns'])>0 for r in group))
            lines.append(f'{label if ratio==10 else ""} & ${ratio}\\%$ & {maximum.pop()} & ' +
                         ' & '.join(f'{c}/{len(group)}' for c in counts) + r' \\')
    lines += old[bottom:]
    return '\n'.join(lines)+'\n'


def literature():
    random,count = load_csvs(SOURCE)
    assert count == 50
    counts = {'Random':[0]*5,'Grid':[0]*5}
    for dataset,_ in DATASETS:
        for ratio in RATIOS:
            rvalues = [random[dataset,ratio][m]['RMSE'].value for m in METHODS]
            grid = read_csv(GRID_SOURCE / f'errors_{dataset}_ratio{ratio}_grid.csv')
            assert len(grid)==7 and {r['Row'] for r in grid}==set(METHODS)
            assert all(r['N_ok']=='1' for r in grid)
            by_method = {r['Row']:r for r in grid}
            gvalues = [Decimal(by_method[m]['RMSE'].split('±')[0].strip()) for m in METHODS]
            assert len(set(gvalues[4:]))==1
            for schema,values in [('Random',rvalues),('Grid',gvalues)]:
                best = min(values)
                for index,value in enumerate(values[:4]): counts[schema][index] += value==best
                counts[schema][4] += min(values[4:])==best
                if schema=='Random':
                    assert all(values[index]!=best for index in [4,6]), 'Revise the TS variant description.'
    text = [line for line in (TABLES/'results_literature_summary.tex').read_text(encoding='utf-8').splitlines()
            if not line.startswith('% Source: current Random/Grid')]
    indices = {'Kriging':0,'IDW':1,'Scattered':2,'NN':3,'TP/TS':4}
    for i,line in enumerate(text):
        method = line.split(' & ')[0]
        if method in indices:
            parts = line.split(' & ')
            parts[1:3] = [f'{counts[s][indices[method]]}/30' for s in ['Random','Grid']]
            text[i] = ' & '.join(parts)
        if line.startswith('IDW & '):
            text[i] = text[i].replace('Ponderação local por distância', 'Ponderação global por distância').replace(
                'com potência 2 e cinco vizinhos fixos.', 'com potência 2 e todas as posições distintas de treinamento.')
        if line.startswith(r'\textit{Nota:}'):
            text[i] = line.replace('antes do arredondamento.',
                'com base nas médias publicadas nos CSVs, incluindo empates nas três casas decimais; '
                'por isso, as contagens podem somar mais de 30.')
    return '% Source: current Random/Grid CSVs; all published-precision ties counted.\n'+'\n'.join(text)+'\n'


def statistical():
    rows = read_csv(REPLAY/'paired_summary.csv')
    assert len(rows)==30 and all(r['n']=='50' and r['csv_summary_matches']=='1' for r in rows)
    lookup = {(r['dataset'],Decimal(r['ratio'])):r for r in rows}
    assert len(lookup)==30
    lines = [
        f'% Source: {(REPLAY / "paired_summary.csv").relative_to(ROOT).as_posix()}.',
        r'\begin{table}[pos=!htbp]',r'\centering',
        r'\caption{Comparação exploratória pareada: TS-I menos Scattered, sob amostragem aleatória.}',
        r'\label{tab:r4-statistical-comparison}',r'\small',r'\begingroup',
        r'\setlength{\tabcolsep}{4pt}',r'\renewcommand{\arraystretch}{1.1}',
        r'\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}ccccc@{}}',r'\toprule',
        r'\textbf{Conjunto} & $\rho$ & $\overline{d}$ & \textbf{IC marginal de 95\%} & $p_{\mathrm{Holm}}$ & \textbf{Vitórias TS-I} \\',r'\midrule',
    ]
    for index,(dataset,label) in enumerate(DATASETS):
        if index: lines.append(r'\addlinespace[3pt]')
        for ratio in RATIOS:
            r = lookup[dataset,Decimal(ratio)/100]
            p = float(r['p_holm'])
            formatted = '$<0.001$' if p<.001 else f'{p:.3f}'
            lines.append(f'{label if ratio==10 else ""} & ${ratio}\\%$ & '+
                         f"${float(r['mean_delta']):+.5f}$ & " +
                         rf"$[{float(r['ci_low']):+.5f},\,{float(r['ci_high']):+.5f}]$ & " +
                         f"{formatted} & {int(r['wins_ts_i'])}/50 " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip',r'\begin{minipage}{\linewidth}',
              r'\footnotesize\raggedright',
              r'\textit{Nota:} $d_k=r_{k,\mathrm{TS-I}}-r_{k,\mathrm{Scattered}}$; diferenças negativas favorecem TS-I. '
              r'Teste bilateral de reamostragem corrigida, com 49 graus de liberdade; ajuste de Holm em uma família de 30 contrastes. '
              r'Os intervalos são marginais, aproximados e sem ajuste de multiplicidade. '
              r'Vitórias contam execuções com $d_k<0$ e são descritivas. '
              r'Diferenças e intervalos estão nas unidades da resposta de cada conjunto (graus Celsius para NOAA). '
              r'A nova reprodução usa coordenadas originais e geometria euclidiana; as médias e os desvios padrão '
              r'dos dois métodos coincidem com os CSVs atuais na precisão publicada. '
              r'A análise é exploratória e não modela explicitamente a dependência espacial.',
              r'\end{minipage}',r'\endgroup',r'\end{table}','']
    return '\n'.join(lines)


def noise_table():
    source = reference_directory('reviewer3') / 'noise_summary.csv'
    rows = read_csv(source)
    assert len(rows)==250 and all(r['n']=='50' and r['fallbacks']=='0' for r in rows)
    lookup = {(r['dataset'],r['method'],Decimal(r['noise_fraction'])):r for r in rows}
    expected = {(d,m,level) for d,_ in DATASETS for m in ['TS-G','TS-I','TS-T','Scattered','B']
                for level in map(Decimal,['0','.01','.05','.1','.2'])}
    assert set(lookup)==expected
    lines = [
        f'% Source: {source.relative_to(ROOT).as_posix()}.',
        r'\begin{table}[pos=!htbp]',r'\centering',
        r'\caption{Sensibilidade a ruído adicional com desvio padrão $0.20s_{Y,\mathrm{train}}$, sob amostragem aleatória com $\rho=30\%$.}',
        r'\label{tab:r3-noise}',r'\small\begingroup',
        r'\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.15}',
        r'\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}ccccccc@{}}',r'\toprule',
        r'Conjunto & \shortstack{TS-I\\$\eta=0$} & \shortstack{TS-I\\$\eta=0.20$} & $D_{\mathrm{TS-I}}$ & $D_B$ & $D_{\mathrm{Scat.}}$ & $D_{\mathrm{TS-G}}$ & $D_{\mathrm{TS-T}}$ \\',r'\midrule',
    ]
    for dataset,label in DATASETS:
        cells = []
        extreme = [lookup[dataset,'TS-I',level] for level in [Decimal(0),Decimal('.2')]]
        best = min(Decimal(f"{float(r['mean_rmse']):.4f}") for r in extreme)
        for r in extreme:
            mean,sd = f"{float(r['mean_rmse']):.4f}",f"{float(r['sd_rmse']):.4f}"
            if Decimal(mean)==best: mean,sd=rf'\mathbf{{{mean}}}',rf'\mathbf{{{sd}}}'
            cells.append(rf'\rmsecell{{{mean}}}{{{sd}}}')
        for method in ['TS-I','B','Scattered','TS-G','TS-T']:
            r = lookup[dataset,method,Decimal('.2')]
            cells.append(f"${float(r['relative_mean_delta_pct']):+.2f}$")
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule\end{tabular*}\endgroup',r'\par\vspace{2pt}\begin{minipage}{\linewidth}\footnotesize',
              r'TS-I: média $\pm$ desvio padrão do RMSE entre 50 partições, com geometria euclidiana nas coordenadas originais; '
              r'unidades da função nos sintéticos e $^\circ\mathrm{C}$ em NOAA. O negrito indica a menor média de TS-I '
              r'entre os dois níveis, incluindo empates nas quatro casas publicadas, sem indicar significância. '
              r'Para cada método, $D=100(\overline r_{0.20}-\overline r_0)/\overline r_0$ é a alteração percentual '
              r'da média; valores positivos indicam piora. Os demais níveis e desvios padrão permanecem nos arquivos '
              r'da análise. São resumos descritivos, sem teste de superioridade.',r'\end{minipage}',r'\end{table}','']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--include-noise',action='store_true')
    args = parser.parse_args()
    grids = grid_data()
    expected = {'grid_occupancy.tex':occupancy(),'reviewer1_grid_limits.tex':limits_table(grids),
                'results_literature_summary.tex':literature(),'statistical_comparison.tex':statistical()}
    if args.include_noise: expected['reviewer3_noise.tex']=noise_table()
    if args.check:
        for name,text in expected.items(): assert (TABLES/name).read_text(encoding='utf-8')==text,name
        print(f'Verified {len(expected)} remaining tables against current CSVs and 1,500 replay grids.')
        return
    print('*** Begin Patch')
    for name,text in expected.items():
        previous = (TABLES/name).read_text(encoding='utf-8')
        if previous==text: continue
        print(f'*** Update File: article/manuscrito/tables/{name}')
        for line in list(difflib.unified_diff(previous.splitlines(),text.splitlines(),n=2,lineterm=''))[2:]:
            print('@@' if line.startswith('@@') else line)
    print('*** End Patch')


if __name__=='__main__':
    main()
