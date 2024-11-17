import json
import os


def escape_html(text):
    """Escape special characters in the given text for HTML."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_text_with_newlines(text):
    if text is None:
        return ""

    lines = text.split("\n")
    formatted_lines = []
    in_code_block = False

    for i, line in enumerate(lines):
        if "[PYTHON]" in line:
            in_code_block = True
            if formatted_lines and formatted_lines[-1].endswith("<br>"):
                formatted_lines[-1] = formatted_lines[-1][:-4]  # Remove the last <br>
            formatted_lines.append("<pre><code>")
            line = line.replace("[PYTHON]", "").strip()
        if "[/PYTHON]" in line:
            in_code_block = False
            line = line.replace("[/PYTHON]", "").strip()
            formatted_lines.append(escape_html(line))
            formatted_lines.append("</code></pre>")
            continue

        if in_code_block:
            formatted_lines.append(escape_html(line) + "\n")
        else:
            formatted_lines.append(escape_html(line) + "<br>")

    return "".join(formatted_lines)


def write_html(task_summaries, output_file):
    """
    Write all task summaries to a single HTML file with toggle buttons.

    Args:
    task_summaries (dict): Contains task information and model results for all tasks
    output_file (str): Path to save the HTML file
    """

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ZeroEval Task Summary</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f0f0f0; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            .correct { background-color: #90EE90; }
            .incorrect { background-color: #FFB6C1; }
            table { border-collapse: collapse; width: 100%; font-size: 14px; margin-top: 20px; display: none; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #f2f2f2; }
            .model-name { cursor: pointer; color: #0066cc; }
            .reasoning { display: none; }
            .reasoning td { background-color: #f9f9f9; }
            h1 { font-size: 28px; color: #333; margin-bottom: 20px; }
            h2 { font-size: 22px; color: #444; margin-top: 30px; margin-bottom: 15px; }
            p { margin: 10px 0; line-height: 1.6; }
            .task-content { display: none; }
            .task-button { margin-right: 10px; padding: 10px 20px; background-color: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            .task-button:hover { background-color: #2980b9; }
            .table-button { margin-top: 10px; padding: 5px 10px; background-color: #2ecc71; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
            .table-button:hover { background-color: #27ae60; }
            tr, th, td { padding: 8px; line-height: 1.2; }
        </style>
        <script>
            function toggleReasoning(modelId) {
                var reasoning = document.getElementById(modelId);
                if (reasoning.style.display === "table-row") {
                    reasoning.style.display = "none";
                } else {
                    reasoning.style.display = "table-row";
                }
            }
            function showTask(taskName) {
                var tasks = document.getElementsByClassName('task-content');
                for (var i = 0; i < tasks.length; i++) {
                    tasks[i].style.display = 'none';
                }
                document.getElementById(taskName).style.display = 'block';
                
                // Update active button style
                var buttons = document.getElementsByClassName('task-button');
                for (var i = 0; i < buttons.length; i++) {
                    buttons[i].style.backgroundColor = '#3498db';
                }
                event.target.style.backgroundColor = '#2980b9';
            }
            function toggleTable(tableId) {
                var table = document.getElementById(tableId);
                if (table.style.display === "table") {
                    table.style.display = "none";
                } else {
                    table.style.display = "table";
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>ZeroEval: Benchmarking LLMs for Reasoning</h1>
            <div>
    """

    # Add toggle buttons
    for task_name in task_summaries.keys():
        html_content += f'<button class="task-button" onclick="showTask(\'{task_name}\')">{task_name}</button>'

    html_content += "</div>"

    # Add content for each task
    for task_name, task_summary in task_summaries.items():
        html_content += f'<div id="{task_name}" class="task-content">'
        for example_id, example in task_summary.items():
            if example["correct_ratio"] < 0.1:  # Only process examples with correct_ratio < 10%
                table_id = f"table_{task_name}_{example_id}"
                html_content += f"""
                <h2>Example: {example_id}</h2>
                <p><strong>Question:</strong> {format_text_with_newlines(example['question'])}</p>
                <p><strong>Correct Answer:</strong> {format_text_with_newlines(str(example['correct_answer']))}</p>
                <button class="table-button" onclick="toggleTable('{table_id}')">Toggle Results</button>
                <table id="{table_id}">
                    <tr>
                        <th>Model</th>
                        <th>Correct</th>
                        <th>Answer</th>
                    </tr>
                """

                correct_models = example["correct_models"].split("|") if example["correct_models"] else []
                incorrect_models = example["incorrect_models"].split("|") if example["incorrect_models"] else []

                model_id = 0
                for model in correct_models + incorrect_models:
                    answer = example["model_answers"].get(model, "N/A")
                    reasoning = example["reasoning"].get(model, "No reasoning provided")
                    is_correct = model in correct_models
                    unique_model_id = f"model_{task_name}_{example_id}_{model_id}"
                    html_content += f"""
                    <tr class="{'correct' if is_correct else 'incorrect'}">
                        <td><span class="model-name" onclick="toggleReasoning('{unique_model_id}')">{escape_html(model)}</span></td>
                        <td>{'True' if is_correct else 'False'}</td>
                        <td>{format_text_with_newlines(answer)}</td>
                    </tr>
                    <tr id="{unique_model_id}" class="reasoning" style="display: none;">
                        <td colspan="3">
                        <strong>Reasoning:</strong> {format_text_with_newlines(reasoning)}
                        </td>
                    </tr>
                    """
                    model_id += 1

                html_content += "</table>"
        html_content += "</div>"

    html_content += """
        </div>
        <script>
            // Show the first task by default
            window.onload = function() {
                var firstTask = document.querySelector('.task-content');
                var firstButton = document.querySelector('.task-button');
                if (firstTask) {
                    firstTask.style.display = 'block';
                }
                if (firstButton) {
                    firstButton.style.backgroundColor = '#2980b9';
                }
            }
        </script>
    </body>
    </html>
    """

    with open(output_file, "w") as f:
        f.write(html_content)


# Read all JSON files in the current folder and convert to HTML
task_summaries = {}
for task in ["gsm", "math-l5", "crux", "mmlu-redux"]:
    json_file = f"state_of_limit/task_summary_{task}.json"
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            task_summaries[task] = json.load(f)
        print(f"Loaded JSON for {task}")
    else:
        print(f"JSON file for {task} not found")

output_file = "state_of_limit/html/all_tasks.html"
write_html(task_summaries, output_file)
print(f"Generated HTML for all tasks: {output_file}")
