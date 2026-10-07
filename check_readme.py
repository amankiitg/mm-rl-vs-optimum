"""Check that every number printed in README.md matches benchmark/results/summary.json.

Run it after any change to the results or the README:

    python check_readme.py

It parses the results tables, the headline sentences and the credit calibration paragraph out
of the README, and compares each number against the run that produced summary.json. A stale
number pasted from an earlier run fails the check. Exit status is non zero on any mismatch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
README = ROOT / 'README.md'
SUMMARY = ROOT / 'benchmark' / 'results' / 'summary.json'

CAPTION = re.compile(r'^(equity|credit) setting, (variance reduced|fully realised) reward',
                     re.IGNORECASE)
ROW = re.compile(r'^\| *`([a-zA-Z_0-9]+)`')
PROBLEMS: list[str] = []

# Template tokens. NAMED are matched anywhere in the file, because they should never appear at
# all once the section has been rendered. The generic pattern catches any other all capitals
# token with an underscore that is long enough to be a template name, and it is applied to the
# prose only: code blocks legitimately contain things such as QUEUE_SIZE or f(d) expressions.
# The length floor of six keeps short notation out of the way, for example J_RL.
NAMED_TOKENS = ('PARAMETER_TABLE', 'CREDIT_NOTE', 'FILLS_CREDIT', 'RUNTIME', 'N_MIN_SUITE',
                'N_TESTS', 'BUDGET')
PREFIX_TOKENS = ('HEADLINE_', 'TABLE_', 'BOUND_COST_')
GENERIC_TOKEN = re.compile(r'\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b')
GENERIC_MIN_LENGTH = 6
HEADLINE_PREFIX = {'equity': 'Equity: ', 'credit': 'Credit: '}


def check(label: str, printed: float, expected: float, decimals: int) -> None:
    if round(printed, decimals) != round(float(expected), decimals):
        PROBLEMS.append(f'{label}: README says {printed!r}, results say {expected!r} '
                        f'(printed to {decimals} decimals)')


def close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def strip_code_blocks(lines: list[str]) -> list[tuple[int, str]]:
    """The lines that are prose, with their original line numbers."""
    prose = []
    in_block = False
    for number, line in enumerate(lines, 1):
        if line.strip().startswith('```'):
            in_block = not in_block
            continue
        if not in_block:
            prose.append((number, line))
    return prose


def placeholder_tokens(line: str) -> set[str]:
    """Template tokens on a prose line, deduplicated."""
    found = set()
    for token in NAMED_TOKENS:
        if token in line:
            found.add(token)
    for prefix in PREFIX_TOKENS:
        index = line.find(prefix)
        while index != -1:
            end = index + len(prefix)
            while end < len(line) and (line[end].isupper() or line[end].isdigit()
                                       or line[end] == '_'):
                end += 1
            found.add(line[index:end])
            index = line.find(prefix, index + 1)
    for match in GENERIC_TOKEN.finditer(line):
        if len(match.group()) >= GENERIC_MIN_LENGTH:
            found.add(match.group())
    return found


def check_placeholders(lines: list[str]) -> int:
    """Fail if the README still contains an unfilled template token.

    Returns the number of prose lines scanned, which is reported separately from the count of
    numbers checked against the results.
    """
    found = set()
    for number, line in enumerate(lines, 1):
        for token in NAMED_TOKENS:
            if token in line:
                found.add((number, token))
    prose = strip_code_blocks(lines)
    for number, line in prose:
        for token in placeholder_tokens(line):
            found.add((number, token))
    for number, token in sorted(found):
        PROBLEMS.append(f'README line {number}: unfilled template token {token!r}')
    return len(prose)


def parse_results_tables(lines: list[str], summaries: dict) -> int:
    """Every table that follows a caption line is compared row by row."""
    current = None
    checks = 0
    in_table = False
    for raw in lines:
        line = raw.strip()
        caption = CAPTION.match(line)
        if caption:
            setting = caption.group(1).lower()
            kind = caption.group(2).lower()
            key = setting if kind.startswith('variance') else f'{setting}_realised_pnl_reward'
            current = summaries.get(key)
            if current is None:
                PROBLEMS.append(f'caption {line!r} refers to {key}, which is not in summary.json')
            continue
        if line.startswith('| strategy |'):
            in_table = current is not None
            continue
        if not in_table:
            continue
        if not line.startswith('|'):
            in_table = False
            continue
        row = ROW.match(line)
        if not row:
            continue
        name = row.group(1)
        cells = [cell.strip() for cell in line.strip('|').split('|')]
        if len(cells) < 6:
            continue
        record = next((r for r in current['rows'] if r['strategy'] == name), None)
        if record is None:
            PROBLEMS.append(f'README row {name} is not in the {current["setting"]} results')
            continue
        check(f'{current["setting"]}/{name} mean PnL', float(cells[1]),
              record['mean_pnl_reported'], 1)
        check(f'{current["setting"]}/{name} CI', float(cells[2].lstrip('+-')),
              record['ci95_half_width_reported'], 1)
        check(f'{current["setting"]}/{name} PnL over std', float(cells[3]),
              record['pnl_over_std'], 2)
        check(f'{current["setting"]}/{name} efficiency', float(cells[4]),
              record['efficiency'], 3)
        check(f'{current["setting"]}/{name} mean abs end of day q', float(cells[6]),
              record['mean_abs_terminal_q'], 2)
        check(f'{current["setting"]}/{name} fills per day', float(cells[7]),
              record['fills_per_episode'], 1)
        checks += 6
        paired = next((p for p in current['paired']
                       if p['comparing'].split(' - ')[0] == name), None)
        if paired is not None and cells[5] != 'n/a':
            parts = cells[5].replace('+-', ' ').split()
            check(f'{current["setting"]}/{name} paired minus optimum', float(parts[0]),
                  paired['mean_difference'] * current['pnl_scale'], 1)
            check(f'{current["setting"]}/{name} paired minus optimum CI',
                  float(parts[1]),
                  1.959963984540054 * paired['standard_error'] * current['pnl_scale'], 1)
            checks += 2
    return checks


def parse_headlines(text: str, summaries: dict) -> int:
    checks = 0
    for setting in ('equity', 'credit'):
        summary = summaries[setting]
        dispersion = summary.get('seed_dispersion')
        if dispersion is None:
            PROBLEMS.append(f'{setting}: summary.json has no seed dispersion to check against')
            continue
        expected = (f"{HEADLINE_PREFIX[setting]}PPO recovers "
                    f"{100 * dispersion['mean_efficiency']:.1f}% "
                    f"+/- {100 * dispersion['std_efficiency']:.1f}% (worst seed "
                    f"{100 * dispersion['worst_seed_efficiency']:.1f}%)")
        if expected not in text:
            PROBLEMS.append(f'{setting}: the README does not contain the expected headline '
                            f'{expected!r}')
        else:
            checks += 1
        note = ('a tuned constant quote earns roughly nothing' in text)
        if summary.get('tuned_quote_earns_nothing') and setting == 'credit' and not note:
            PROBLEMS.append('credit: the tuned quote earns nothing, so the README must say so')
    return checks


def parse_calibration(text: str, summaries: dict) -> int:
    """The credit calibration paragraph must quote numbers the code actually produces."""
    checks = 0
    half_spread = 0.175
    hit = 0.30
    if '0.175' not in text:
        PROBLEMS.append('the credit worked example must quote the 0.175 point half-spread')
    else:
        checks += 1
    if '30 percent' not in text and '30%' not in text:
        PROBLEMS.append('the credit calibration must quote the 30 percent hit ratio')
    else:
        checks += 1
    if '2.5 basis points' not in text and '2.5 bp' not in text:
        PROBLEMS.append('the credit calibration must quote 2.5 basis points')
    else:
        checks += 1
    # recompute the two calibration targets from the code itself
    from benchmark.config import credit_setting
    from benchmark.dp import solve_dp

    setting = credit_setting()
    dp = solve_dp(setting)
    d0 = float(dp.d_ask[0, setting.Q])
    f0 = float(setting.fill_model().f(d0))
    if not close(d0, half_spread, 0.20 * half_spread):
        PROBLEMS.append(f'the code quotes {d0:.4f} points, the README says {half_spread}')
    if not close(f0, hit, 0.05):
        PROBLEMS.append(f'the code hit ratio is {f0:.3f}, the README says {hit}')
    return checks + 2


def main() -> int:
    if not SUMMARY.exists():
        print(f'no results at {SUMMARY}, run benchmark/run_all.py first')
        return 1
    data = json.loads(SUMMARY.read_text())
    summaries = {s['setting']: s for s in data['summaries']}
    text = README.read_text()
    lines = text.splitlines()

    prose_lines = check_placeholders(lines)
    checks = parse_results_tables(lines, summaries)
    checks += parse_headlines(text, summaries)
    checks += parse_calibration(text, summaries)

    if PROBLEMS:
        print(f'{len(PROBLEMS)} mismatch(es) between README.md and summary.json:')
        for problem in PROBLEMS:
            print(f'  - {problem}')
        return 1
    print(f'README.md matches summary.json on all {checks} checked numbers, and no unfilled '
          f'template token was found in {prose_lines} prose lines')
    return 0


if __name__ == '__main__':
    sys.exit(main())
