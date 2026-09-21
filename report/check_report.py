"""
check_report.py - Static checks on report/main.tex (no LaTeX install needed).

    python report/check_report.py

1. Every macro used is defined in generated/numbers.tex (or is standard LaTeX).
2. Every \\cite key has a \\bibitem, and every \\bibitem is cited.
3. Braces and \\begin/\\end environments balance.
4. Lists every digit typed by hand in the body text, so each can be confirmed
   to be a method setting or citation detail rather than a result.
5. Prints a word count per section for the page budget.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
tex = (HERE / 'main.tex').read_text(encoding='utf-8')
nums = (HERE / 'generated/numbers.tex').read_text(encoding='utf-8')
defined = set(re.findall(r'\\newcommand\{\\(\w+)\}', nums))

body = re.sub(r'(?<!\\)%.*', '', tex)                     # strip comments
doc = body[body.index(r'\begin{document}'):]
main_text = doc[:doc.index(r'\begin{thebibliography}')]

problems = []

# 1. macros
used = set(re.findall(r'\\([A-Za-z]+)\{\}', main_text))
missing = sorted(u for u in used if u not in defined)
if missing:
    problems.append(f'undefined result macros: {missing}')

# 2. citations
cites = set()
for grp in re.findall(r'\\cite\{([^}]*)\}', main_text):
    cites.update(k.strip() for k in grp.split(','))
items = re.findall(r'\\bibitem\{([^}]*)\}', doc)
if len(items) != len(set(items)):
    problems.append('duplicate bibitem keys')
if cites - set(items):
    problems.append(f'cited but not in bibliography: {sorted(cites - set(items))}')
if set(items) - cites:
    problems.append(f'in bibliography but never cited: {sorted(set(items) - cites)}')

# 3. balance ("\\" is a line break, so drop it first: otherwise the "\}" in
#    "2026\\}" would be misread as an escaped brace)
b3 = body.replace('\\\\', ' ')
if b3.count('{') - b3.count(r'\{') != b3.count('}') - b3.count(r'\}'):
    problems.append('unbalanced braces')
envs = re.findall(r'\\(begin|end)\{(\w+\*?)\}', body)
stack = []
for kind, name in envs:
    if kind == 'begin':
        stack.append(name)
    elif not stack or stack.pop() != name:
        problems.append(f'environment mismatch at \\end{{{name}}}')
        break
if stack:
    problems.append(f'unclosed environments: {stack}')

# 4. hand-typed digits in prose (skip tikz coordinates, labels, refs, macros)
prose = main_text[main_text.index(r'\section'):]
prose = re.sub(r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}', ' ', prose, flags=re.S)
prose = re.sub(r'\\(label|ref|cite|input|includegraphics|setlength|tabcolsep)(\[[^]]*\])?\{[^}]*\}', ' ', prose)
prose = re.sub(r'\\begin\{tabularx\}\{[^}]*\}\{[^}]*\}', ' ', prose)
prose = re.sub(r'\[width=[^]]*\]|\[[hbt]+\]', ' ', prose)
hand = []
for m in re.finditer(r'[^\s{}]*\d[^\s{}]*', prose):
    ctx = prose[max(0, m.start() - 35):m.end() + 25].replace('\n', ' ')
    hand.append((m.group(0), ctx))

# 5. word counts
secs = re.split(r'\\section\*?\{([^}]*)\}', main_text)
words = {}
for i in range(1, len(secs), 2):
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^]]*\])?', ' ', secs[i + 1])
    t = re.sub(r'[{}$\\&~]', ' ', t)
    words[secs[i]] = len([w for w in t.split() if re.search(r'[A-Za-z]', w)])

print('== problems ==')
print('\n'.join(problems) if problems else 'none')
print(f'\n== result macros used: {len(used & defined)} / defined: {len(defined)} ==')
print(f'== citations: {len(cites)} cited, {len(items)} bibitems ==')
print('\n== hand-typed digits in prose (confirm none is a result) ==')
for tok, ctx in hand:
    print(f'  {tok:18s} ...{ctx}...')
print('\n== words per section (approx.) ==')
for k, v in words.items():
    print(f'  {v:5d}  {k}')
print(f'  {sum(words.values()):5d}  TOTAL (excluding tables, figures, references)')
