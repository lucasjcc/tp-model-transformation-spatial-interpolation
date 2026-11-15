"""Export manuscript RMSE/rule tables from the published CSV means and SDs.

Only the Random tables are updated. Ties at the CSV precision are all bold;
MAT files and unpublished digits are deliberately not used for ranking.
Run with --check to independently parse and verify the existing LaTeX cells,
or --patch to emit an apply_patch patch without modifying the tables.
"""

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal
import difflib
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "article/manuscrito/tables"


def reference_directory(category, run_tag=""):
    """Resolve read-only CNO references, preferring the current output layout."""
    current = ROOT / "results" / category / "cno" / run_tag
    required = {
        "reviewer1": "ablation_summary.csv",
        "reviewer3": "noise_summary.csv",
        "reviewer4": "paired_summary.csv",
    }
    def has_records(directory):
        if category == "monte_carlo":
            sampling = run_tag.split("_n", 1)[0]
            return any(directory.glob(f"errors_*_{sampling}.csv"))
        return (directory / required[category]).is_file()

    if has_records(current):
        return current
    archives = [
        directory for directory in (ROOT / "results" / category).glob("*/cno")
        if has_records(directory)
    ]
    if len(archives) > 1:
        raise ValueError("Multiple archived reference folders were found.")
    return archives[0] if archives else current


SOURCE = reference_directory("monte_carlo", "random_n50")
GRID_SOURCE = reference_directory("monte_carlo", "grid_n1")
DATASETS = [
    ("franke", "Franke"),
    ("peaks", "Peaks"),
    ("sinc", "Sinc"),
    ("gauss_mixture", "Mistura gaussiana"),
    ("wind_temperature_month01", "NOAA (janeiro)"),
    ("wind_temperature_month02", "NOAA (fevereiro)"),
    ("wind_temperature_month03", "NOAA (março)"),
    ("wind_temperature_month04", "NOAA (abril)"),
    ("wind_temperature_month05", "NOAA (maio)"),
    ("wind_temperature_month06", "NOAA (junho)"),
]
RATIOS = (10, 30, 50)
METHODS = (
    "Kriging", "IDW", "Scattered", "Neural Net",
    "TS (Global Mean)", "TS (Interp)", "TS (Tensor Comp)",
)
PAIR = re.compile(r"(\d+\.\d{3})\s*±\s*(\d+\.\d{3})")
LATEX_NUMBER = r"(?:\\mathbf\{\d+\.\d{3}\}|\d+\.\d{3})"
LATEX_CELL = re.compile(
    rf"\\rmsecell\{{({LATEX_NUMBER})\}}\{{({LATEX_NUMBER})\}}"
)


@dataclass(frozen=True)
class Metric:
    mean: str
    sd: str

    @property
    def value(self):
        return Decimal(self.mean)


def parse_metric(text, location):
    match = PAIR.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"Missing/invalid three-decimal mean and SD: {location}: {text!r}")
    return Metric(*match.groups())


def load_csvs(source):
    data = {}
    sample_counts = set()
    expected = set()
    for dataset, _ in DATASETS:
        for ratio in RATIOS:
            name = f"errors_{dataset}_ratio{ratio}_random.csv"
            expected.add(name)
            path = source / name
            with path.open(encoding="utf-8-sig", newline="") as stream:
                records = list(csv.DictReader(stream))
            if len(records) != len(METHODS) or {r["Row"] for r in records} != set(METHODS):
                raise ValueError(f"Missing, duplicate or unexpected method rows: {path}")
            lookup = {}
            for row in records:
                method = row["Row"]
                count = int(row["N_ok"])
                if count < 2:
                    raise ValueError(f"Sample SD requires at least two valid repetitions: {path}, {method}")
                sample_counts.add(count)
                rmse = parse_metric(row["RMSE"], f"{name}, {method}, RMSE")
                rules = None
                if method in METHODS[4:]:
                    rules = parse_metric(row["Rules"], f"{name}, {method}, Rules")
                    if rules.value < 1:
                        raise ValueError(f"Invalid TS rule count: {path}, {method}")
                elif row["Rules"] != "N/A":
                    raise ValueError(f"Unexpected rule count for a non-TS method: {path}, {method}")
                lookup[method] = {"RMSE": rmse, "Rules": rules}
            data[dataset, ratio] = lookup
    if len(sample_counts) != 1:
        raise ValueError(f"Unequal numbers of valid repetitions: {sorted(sample_counts)}")
    if {p.name for p in source.glob("errors_*_random.csv")} != expected:
        raise ValueError("Unexpected/missing Random configurations in the selected source directory")
    return data, sample_counts.pop()


def cell(metric, best):
    mean, sd = metric.mean, metric.sd
    if metric.value == best:
        mean, sd = rf"\mathbf{{{mean}}}", rf"\mathbf{{{sd}}}"
    return rf"\rmsecell{{{mean}}}{{{sd}}}"


def rmse_table(data, ratio, source_label, count):
    lines = [
        f"% Source: {source_label}/errors_*_ratio{ratio}_random.csv.",
        "% Ranking uses CSV means; all ties at the published precision are bold.",
        r"\begin{table}[pos=!htbp]", r"\centering",
        rf"\caption{{RMSE sob amostragem aleatória com ${ratio}\%$ dos dados para treinamento.}}",
        rf"\label{{tab:results_random_0{ratio // 10}}}",
        r"\begingroup", r"\small", r"\setlength{\tabcolsep}{1.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}ccccccc@{}}",
        r"    \toprule",
        r"    & \multicolumn{4}{c}{\textbf{Métodos de referência}} & \multicolumn{3}{c}{\textbf{Modelos TS}} \\",
        r"    \cmidrule(lr){2-5} \cmidrule(lr){6-8}",
        r"    \textbf{Conjunto de dados} & \textbf{Kriging} & \textbf{IDW} & \textbf{Scattered} & \textbf{NN} & \textbf{TS-G} & \textbf{TS-I} & \textbf{TS-T} \\",
        r"    \midrule",
    ]
    for index, (dataset, label) in enumerate(DATASETS):
        if index == 4:
            lines[-1] = r"    \midrule"
        values = [data[dataset, ratio][method]["RMSE"] for method in METHODS]
        best = min(value.value for value in values)
        lines.append("    " + label + " & " + " & ".join(cell(value, best) for value in values) + r" \\")
        lines.append(r"    \addlinespace[2pt]")
    lines += [r"    \bottomrule", r"\end{tabular*}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}", r"\footnotesize\raggedright",
              r"\textit{Nota:} média $\pm$ desvio padrão amostral, em " + str(count) +
              r" execuções válidas por conjunto e método. Valores transcritos da coluna \texttt{RMSE} dos CSVs, "
              r"com três casas decimais. O negrito destaca a menor média publicada em cada linha, incluindo "
              r"todos os empates nessa precisão, sem indicar significância estatística ou menor desvio padrão. "
              r"Kriging: krigagem; IDW: ponderação global pelo inverso da distância; Scattered: vizinhos naturais "
              r"nos conjuntos sintéticos e interpolação linear nos dados NOAA, ambos em geometria euclidiana; "
              r"NN: rede neural. TS-G, TS-I e TS-T: transformação TP com convexificação CNO e imputação pela "
              r"média global, interpolação espacial e completamento tensorial, respectivamente. As escalas "
              r"de resposta diferem entre conjuntos; os RMSEs devem ser comparados dentro de cada linha.",
              r"\end{minipage}", r"\endgroup", r"\end{table}", ""]
    return "\n".join(lines)


def rules_table(data, source_label, count):
    lines = [
        f"% Source: Rules column in {source_label}/errors_*_random.csv.",
        "% Ranking uses CSV means; all ties at the published precision are bold.",
        r"\begin{table}[p]", r"\centering",
        r"\caption{Número de regras dos modelos TS com convexificação CNO sob amostragem aleatória.}",
        r"\label{tab:rules_random}", r"\begingroup", r"\small",
        r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.08}",
        r"\begin{tabular*}{\linewidth}{@{}l@{\extracolsep{\fill}}cccc@{}}",
        r"    \toprule",
        r"    \textbf{Conjunto de dados} & \textbf{Treinamento} & \textbf{TS-G} & \textbf{TS-I} & \textbf{TS-T} \\",
        r"    \midrule",
    ]
    for index, (dataset, label) in enumerate(DATASETS):
        if index:
            lines.append(r"    \midrule")
        for ratio in RATIOS:
            prefix = rf"\multirow{{3}}{{*}}{{{label}}}" if ratio == RATIOS[0] else ""
            values = [data[dataset, ratio][method]["Rules"] for method in METHODS[4:]]
            best = min(value.value for value in values)
            lines.append("    " + prefix + rf" & ${ratio}\%$ & " +
                         " & ".join(cell(value, best) for value in values) + r" \\")
    lines += [r"    \bottomrule", r"\end{tabular*}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}", r"\footnotesize\raggedright",
              r"\textit{Nota:} média $\pm$ desvio padrão amostral do número de regras em " + str(count) +
              r" execuções por configuração. Valores transcritos da coluna \texttt{Rules}, com as três casas "
              r"decimais exportadas; as médias podem ser fracionárias, embora cada modelo tenha um número "
              r"inteiro de regras. O negrito destaca a menor média publicada entre TS-G, TS-I e TS-T em cada "
              r"linha, incluindo todos os empates nessa precisão, sem comparar os desvios padrão. TS-G, TS-I "
              r"e TS-T utilizam imputação pela média global, interpolação espacial e completamento tensorial, "
              r"respectivamente. Kriging, IDW, Scattered e NN não são modelos de regras TS e apresentam "
              r"\texttt{N/A} nessa coluna dos resultados; não lhes foi atribuída uma contagem de regras.",
              r"\end{minipage}", r"\endgroup", r"\end{table}", ""]
    return "\n".join(lines)


def check_cells(line, metrics):
    """Independently decode LaTeX numbers and bold flags, then compare to CSV."""
    matches = LATEX_CELL.findall(line)
    if len(matches) != len(metrics):
        raise ValueError(f"Wrong number of LaTeX cells: {line}")
    minimum = min(Decimal(metric.mean) for metric in metrics)
    for (mean_token, sd_token), metric in zip(matches, metrics):
        mean_bold = mean_token.startswith(r"\mathbf{")
        sd_bold = sd_token.startswith(r"\mathbf{")
        mean_text = mean_token[8:-1] if mean_bold else mean_token
        sd_text = sd_token[8:-1] if sd_bold else sd_token
        if (mean_text, sd_text) != (metric.mean, metric.sd):
            raise ValueError(f"LaTeX/CSV mismatch: {(mean_text, sd_text)} != {metric}")
        expected_bold = Decimal(mean_text) == minimum
        if mean_bold != expected_bold or sd_bold != expected_bold:
            raise ValueError(f"Incorrect minimum/tie highlighting: {line}")


def check_tables(data, texts):
    for ratio in RATIOS:
        name = f"results_random_0{ratio // 10}.tex"
        rows = [line.strip() for line in texts[name].splitlines() if LATEX_CELL.search(line)]
        if len(rows) != len(DATASETS):
            raise ValueError(f"Wrong dataset row count in {name}")
        for line, (dataset, label) in zip(rows, DATASETS):
            if not line.startswith(label + " & "):
                raise ValueError(f"Wrong dataset order/label in {name}: {line}")
            check_cells(line, [data[dataset, ratio][method]["RMSE"] for method in METHODS])
    rows = [line.strip() for line in texts["rules_random.tex"].splitlines() if LATEX_CELL.search(line)]
    if len(rows) != len(DATASETS) * len(RATIOS):
        raise ValueError("Wrong rule-table row count")
    for row_index, ((dataset, label), ratio) in enumerate(
            (case, ratio) for case in DATASETS for ratio in RATIOS):
        line = rows[row_index]
        expected_label = rf"\multirow{{3}}{{*}}{{{label}}}" if ratio == RATIOS[0] else ""
        expected_prefix = expected_label + rf" & ${ratio}\%$ & "
        if not line.startswith(expected_prefix.strip()):
            raise ValueError(f"Wrong dataset/ratio in rule table: {line}")
        check_cells(line, [data[dataset, ratio][method]["Rules"] for method in METHODS[4:]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--patch", action="store_true")
    args = parser.parse_args()
    data, count = load_csvs(SOURCE)
    label = SOURCE.relative_to(ROOT).as_posix()
    texts = {f"results_random_0{ratio // 10}.tex": rmse_table(data, ratio, label, count)
             for ratio in RATIOS}
    texts["rules_random.tex"] = rules_table(data, label, count)
    check_tables(data, texts)
    if args.check:
        existing = {name: (TABLES / name).read_text(encoding="utf-8") for name in texts}
        check_tables(data, existing)
        for name, expected in texts.items():
            if existing[name] != expected:
                raise ValueError(f"Outdated table notes, source or formatting: {name}")
        print(f"Verified 30 CSVs, 210 RMSE pairs, 90 rule-count pairs and all bold minima/ties; N_ok={count}.")
    elif args.patch:
        print("*** Begin Patch")
        for name, text in texts.items():
            previous = (TABLES / name).read_text(encoding="utf-8")
            if previous == text:
                continue
            print(f"*** Update File: article/manuscrito/tables/{name}")
            diff = list(difflib.unified_diff(previous.splitlines(), text.splitlines(),
                                             n=2, lineterm=""))
            for line in diff[2:]:
                print("@@" if line.startswith("@@") else line)
        print("*** End Patch")
    else:
        for name, text in texts.items():
            (TABLES / name).write_text(text, encoding="utf-8")
        check_tables(data, texts)
        print(f"Updated four tables from 30 CSVs; N_ok={count}. Run again with --check to audit the saved files.")


if __name__ == "__main__":
    main()
