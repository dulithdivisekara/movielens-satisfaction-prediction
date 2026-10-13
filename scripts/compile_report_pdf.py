"""
Generate Group_08.pdf from FINAL_GROUP_REPORT.md using WeasyPrint
"""

import weasyprint

html_doc = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Group 08 - Final Report</title>
<style>
@page { size: A4; margin: 18mm; @bottom-right { content: counter(page); font-size: 9pt; } }
body { font-family: 'Helvetica Neue', Arial, sans-serif; line-height: 1.55; font-size: 10.5pt; color: #2d3748; }
h1 { color: #1a365d; font-size: 18pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 4px; margin-top: 24pt; }
h2 { color: #2b6cb0; font-size: 14pt; margin-top: 18pt; border-bottom: 1px solid #e2e8f0; padding-bottom: 3px; }
h3 { color: #2d3748; font-size: 11.5pt; margin-top: 12pt; }
h4 { color: #4a5568; font-size: 10.5pt; margin-top: 10pt; }
table { width: 100%; border-collapse: collapse; margin: 12pt 0; font-size: 9pt; }
th, td { border: 1px solid #cbd5e0; padding: 5pt 7pt; text-align: left; }
th { background-color: #edf2f7; color: #1a202c; font-weight: bold; }
code { background-color: #f7fafc; padding: 1pt 3pt; border-radius: 3pt; font-family: monospace; font-size: 9pt; }
pre { background-color: #f7fafc; border: 1px solid #e2e8f0; padding: 8pt; border-radius: 4pt; font-size: 8.5pt; overflow-x: auto; }
ul, ol { margin-top: 4pt; margin-bottom: 8pt; padding-left: 18pt; }
li { margin-bottom: 3pt; }
p { margin-top: 4pt; margin-bottom: 6pt; }
</style>
</head>
<body>
"""

with open('FINAL_GROUP_REPORT.md', 'r', encoding='utf-8') as f:
    text = f.read()

lines = text.split('\n')
in_table = False
html_body = []
for line in lines:
    sline = line.strip()
    if sline.startswith('# '):
        html_body.append(f"<h1>{sline[2:]}</h1>")
    elif sline.startswith('## '):
        html_body.append(f"<h2>{sline[3:]}</h2>")
    elif sline.startswith('### '):
        html_body.append(f"<h3>{sline[4:]}</h3>")
    elif sline.startswith('#### '):
        html_body.append(f"<h4>{sline[5:]}</h4>")
    elif sline.startswith('|') and '|' in sline[1:]:
        if not in_table:
            html_body.append("<table>")
            in_table = True
        cells = [c.strip() for c in sline.split('|')[1:-1]]
        if cells and '---' in cells[0]:
            continue
        is_header = ('Member' in cells[0] or 'PARADIGM' in cells[0] or 'Tool Name' in cells[0] or 'Student' in cells[0])
        tag = 'th' if is_header else 'td'
        row_html = ''.join([f"<{tag}>{c}</{tag}>" for c in cells])
        html_body.append(f"<tr>{row_html}</tr>")
    else:
        if in_table:
            html_body.append("</table>")
            in_table = False
        if sline.startswith('- ') or sline.startswith('* '):
            html_body.append(f"<li>{sline[2:]}</li>")
        elif sline.startswith('1. ') or sline.startswith('2. ') or sline.startswith('3. ') or sline.startswith('4. '):
            html_body.append(f"<li><b>{sline[:3]}</b> {sline[3:]}</li>")
        elif sline == '' or sline == '---':
            pass
        elif not sline.startswith('```'):
            html_body.append(f"<p>{sline}</p>")

if in_table:
    html_body.append("</table>")

full_html = html_doc + '\n'.join(html_body) + "</body></html>"
weasyprint.HTML(string=full_html).write_pdf('Group_08.pdf')
print("Successfully generated Group_08.pdf ready for submission!")
