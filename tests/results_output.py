"""
Created on Fri Nov 29 13:24:00 2024


@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os
from datetime import datetime

import yaml

from tests.assertions import calculate_error


def results_files_to_markdown(results_dir: str, output_file: str):
    with open(output_file, 'w') as f:
        yml_files = [f for f in os.listdir(results_dir) if f.endswith('.yml')]
        
        f.write("# Summary of test results\n")
        f.write(f"**Date: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}**\n\n")
        f.write(f"**Number of projects: {len(yml_files)}**\n\n")
        f.write("\n")
        for file in yml_files:
            with open(os.path.join(results_dir, file), 'r') as yml_file:
                data: dict = yaml.safe_load(yml_file)
                model = data['model']
                ue_distribution = data['ue_distribution']
                project = f"{model}_{ue_distribution}"
                run_time = data['run_time']
                
                f.write(f"## {project} results\n")
                f.write(f"**Run time**: {run_time}\n")
                
                # Get all the other keys that have not been printed
                keys: list[str] = [k for k in data.keys() if k not in ['model', 'ue_distribution', 'run_time']]
                for key in keys:
                    f.write(f"## {key}\n")
                    f.write("Attribute | Expected | Actual | Difference | Success |\n")
                    f.write("|-----------|----------|--------|------------|---------|\n")
                    values: dict = data[key]
                    successes: dict = values["successes"]
                    failures: dict = values["failures"]
                    
                    results: dict = {}
                    if successes is not None:
                        results.update(successes)
                    if failures is not None:
                        results.update(failures)

                    for operation, value in results.items():
                        expected: float = float(value['expected'])
                        actual: float = float(value['actual'])
                        success = '<span style="color:green">yes</span>' if operation in successes.keys() else '<span style="color:red">no</span>'
                        f.write(f"| {operation} | {expected} | {actual} | {abs(actual - expected)} | {success} |\n")
    print(f'Generated Markdown report in: {os.path.realpath(output_file)}')


def results_files_to_html(results_dir: str, output_file: str):
    with open(output_file, 'w', encoding="utf-8") as f:
        yml_files = [f for f in os.listdir(results_dir) if f.endswith('.yml')]
        
        f.write("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Giulia Tests Summary</title>
            
            <style>
            .title { cursor: pointer; }
            </style>
        </head>
        <body>
        """)
        
        f.write(f'<p class="summary">Date: <span>{datetime.today().strftime("%Y-%m-%d %H:%M:%S")}</span></p>\n')
        f.write(f'<p class="summary">Number of Projects: <span>{len(yml_files)}</span></p>\n')

        img_path = os.path.join(results_dir, f'usage.png')
        if os.path.exists(img_path):
            f.write(
                f'<br/><a href="./tests/usage.csv" title="Click to download CSV"><img style="width: 700px;" src="./tests/usage.png" /></a>\n')
        else:
            print('General usage plot does not exist:', img_path)

        for file in yml_files:
            with open(os.path.join(results_dir, file), 'r') as yml_file:
                data: dict = yaml.safe_load(yml_file)
                model = data['model']
                ue_distribution = data['ue_distribution']
                project = f"{model}_{ue_distribution}"
                run_time = data['run_time']
                
                f.write(f"<h1 class=\"title\" id=\"{project}\"><span>▼</span> {project} results</h1>\n")
                f.write(f'<div id=\"{project}_container\">\n')
                f.write(f'<p class="summary">Run time: <span>{run_time}</span>\n')

                usage_filename = f'usage_{model}_{ue_distribution}'
                img_path = os.path.join(results_dir, f'{usage_filename}.png')
                if os.path.exists(img_path):
                    f.write(f'<br/><a href="./tests/{usage_filename}.csv" title="Click to download CSV"><img style="width: 700px;" src="./tests/{usage_filename}.png" /></a>\n')
                else:
                    print('Usage plot does not exist:', img_path)
                
                # Get all the other keys that have not been printed
                keys: list[str] = [k for k in data.keys() if k not in ['model', 'ue_distribution', 'run_time']]
                for key in keys:
                    f.write(f"<h2>{key}</h2>\n")
                    f.write("<table>\n")
                    f.write("<tr><th>Attribute</th><th>Expected</th><th>Actual</th><th>Error %</th><th>Success</th></tr>\n")
                    values: dict = data[key]
                    successes: dict = values["successes"]
                    failures: dict = values["failures"]
                    
                    results: dict = {}
                    if successes is not None:
                        results.update(successes)
                    if failures is not None:
                        results.update(failures)

                    for operation, value in results.items():
                        expected: float = float(value['expected'])
                        actual: float = float(value['actual'])
                        success = '<span style="color:green">yes</span>' if operation in successes.keys() else '<span style="color:red">no</span>'
                        err = calculate_error(actual, expected)
                        f.write(f"<tr><td>{operation}</td><td>{expected}</td><td>{actual}</td><td>{err}</td><td>{success}</td></tr>\n")
                    f.write("</table>\n")
                f.write(f'</div>\n')

        f.write("""
        <script>
        window.addEventListener('load', function () {
            const titles = document.getElementsByClassName('title');
            for (const title of titles) {
                title.addEventListener('click', function () {
                    const collapsed = title.getAttribute('data-collapsed') !== 'false';
                    const container = document.getElementById(title.id + '_container');
                    const arrow = title.children[0];
                    title.setAttribute('data-collapsed', !collapsed);
                    container.style.display = collapsed ? 'none' : null;
                    arrow.innerText = collapsed ? '►' : '▼';
                });
            }
        });
        </script>
        </body>
        </html>
        """)
    print(f'Generated HTML report in: {os.path.realpath(output_file)}')
