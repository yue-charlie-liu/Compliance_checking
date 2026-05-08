import json
import numpy as np
from pathlib import Path
from typing import Any
import warnings

warnings.filterwarnings('ignore')


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_id(title: str) -> str:
    """Extract a stable ID from a title string."""
    if not title:
        return ""
    id_part = ""
    for char in title:
        if char.isalnum() or char in '.-':
            id_part += char
        elif char == ' ':
            break
        else:
            break
    return id_part.strip() or title[:20]


def create_interactive_heatmap() -> None:
    """Create an interactive Plotly heatmap from JSON inputs."""
    base_dir = Path(__file__).resolve().parent
    check_results_file = base_dir / "result" / "check_results.json"
    whs_regulations_file = base_dir.parent / "chunking" / "flattened" / "model WHS regulations.json"
    output_file = base_dir / "result" / "compliance_heatmap.html"

    print(f"Loading check results from {check_results_file}")
    check_results = load_json(check_results_file)

    print(f"Loading WHS regulations from {whs_regulations_file}")
    whs_regulations = load_json(whs_regulations_file)

    relevant_provision_titles = []
    relevant_provision_ids = []
    relevant_provision_set = set()
    provision_titles = []
    provision_ids = []
    provision_index = {}

    relevant_source_titles = []
    relevant_source_ids = []
    relevant_source_set = set()

    for item in check_results:
        source_title = item.get("source_title", "").strip()
        if not source_title:
            continue
        matches = item.get("matches", [])
        for match in matches:
            target_title = match.get("target_title", "").strip()
            if not target_title:
                continue
            compliance = match.get("compliance", {})
            if compliance.get("is_relevant") is True and compliance.get("compliant") is not None:
                relevant_source_set.add(source_title)
                relevant_provision_set.add(target_title)

    for item in whs_regulations:
        title = item.get("title", "").strip()
        if not title or title not in relevant_provision_set:
            continue
        relevant_provision_ids.append(extract_id(title))
        relevant_provision_titles.append(title)
        provision_index[title] = len(relevant_provision_ids) - 1

    for item in check_results:
        title = item.get("source_title", "").strip()
        if not title or title not in relevant_source_set:
            continue
        relevant_source_ids.append(extract_id(title))
        relevant_source_titles.append(title)

    source_ids = relevant_source_ids
    source_titles = relevant_source_titles
    provision_ids = relevant_provision_ids
    provision_titles = relevant_provision_titles
    source_index = {title: idx for idx, title in enumerate(source_titles)}

    print(f"Found {len(source_titles)} relevant source titles")
    print(f"Found {len(provision_titles)} relevant provision titles")
    print(f"Creating interactive {len(source_titles)} x {len(provision_titles)} heatmap")

    heatmap_data = np.zeros((len(source_titles), len(provision_titles)), dtype=int)
    hover_texts = [["" for _ in range(len(provision_titles))] for _ in range(len(source_titles))]
    stats = {"compliant": 0, "non_compliant": 0, "not_relevant": 0}

    for item in check_results:
        source_title = item.get("source_title", "").strip()
        if source_title not in source_index:
            continue
        source_idx = source_index[source_title]
        matches = item.get("matches", [])
        for match in matches:
            target_title = match.get("target_title", "").strip()
            if target_title not in provision_index:
                continue
            target_idx = provision_index[target_title]
            compliance = match.get("compliance", {})
            is_relevant = compliance.get("is_relevant", False)
            compliant = compliance.get("compliant", None)
            reasoning = compliance.get("reasoning", "")
            remediation = compliance.get("remediation", "")
            match_score = match.get("match_score", 0.0)

            if is_relevant is True and compliant is True:
                value = 2
                status = "COMPLIANT"
                stats["compliant"] += 1
            elif is_relevant is True and compliant is False:
                value = 1
                status = "NON-COMPLIANT"
                stats["non_compliant"] += 1
            else:
                value = 0
                status = "Not Relevant"
                stats["not_relevant"] += 1

            heatmap_data[source_idx, target_idx] = value
            if is_relevant is True and compliant is not None:
                hover_lines = [
                    f"<b>Source ID:</b> {source_ids[source_idx]}",
                    f"<b>Provision ID:</b> {provision_ids[target_idx]}",
                    f"<b>Status:</b> {status}",
                    f"<b>Relevance:</b> {'Yes' if is_relevant else 'No'}",
                    f"<b>Match Score:</b> {match_score:.3f}",
                ]
                hover_lines.append(f"<b>Compliant:</b> {'Yes' if compliant else 'No'}")
                if reasoning:
                    hover_lines.append(f"<br><b>Reasoning:</b> {wrap_text(reasoning, 70)}")
                if remediation:
                    hover_lines.append(f"<br><b>Remediation:</b> {wrap_text(remediation, 70)}")

                hover_texts[source_idx][target_idx] = "<br>".join(hover_lines)

            else:
                # 灰色（不相关）Cell 文本直接设为空
                hover_texts[source_idx][target_idx] = ""

    print(f"\nCompliance Statistics:")
    print(f"  - Compliant (Green): {stats['compliant']}")
    print(f"  - Non-compliant (Red): {stats['non_compliant']}")
    # print(f"  - Not Relevant (Grey): {stats['not_relevant']}")

    html_content = generate_html(
        heatmap_data.tolist(),
        source_ids,
        provision_ids,
        hover_texts,
        stats,
        len(source_titles),
        len(provision_titles),
    )

    print(f"\nSaving interactive heatmap to {output_file}")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html_content, encoding='utf-8')
    print(f"✓ Interactive heatmap saved to {output_file}")

def wrap_text(text, width=70):
    if not text: return ""

    if isinstance(text, list):
        text = " ".join([str(i) for i in text])
    
    text = str(text)

    import textwrap
    return "<br>".join(textwrap.wrap(text, width=width))

def generate_html(
    heatmap_data,
    source_ids,
    provision_ids,
    hover_texts,
    stats,
    source_count,
    provision_count,
) -> str:
    # 颜色定义
    color_map = {
        0: '#e0e0e0',  # Not Relevant (Grey)
        1: '#ff6b6b',  # Non-Compliant (Red)
        2: '#51cf66'   # Compliant (Green)
    }

    # 将 Heatmap 矩阵转换为 Scatter 点集，方便控制形状
    scatter_x = []
    scatter_y = []
    scatter_colors = []
    scatter_text = []
    
    for r in range(len(source_ids)):
        for c in range(len(provision_ids)):
            scatter_x.append(c)
            scatter_y.append(len(source_ids) - 1 - r) # 倒序排列，让第一行在最上面
            scatter_colors.append(color_map[heatmap_data[r][c]])
            scatter_text.append(hover_texts[r][c])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Compliance Policy Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: 'Inter', system-ui, sans-serif; background: #f8f9fa; padding: 40px; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); padding: 40px; }}
        .header {{ margin-bottom: 40px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
        .header h1 {{ color: #1a1a1a; font-size: 24px; }}
        .stats-bar {{ display: flex; gap: 20px; margin-bottom: 30px; }}
        .stat-card {{ flex: 1; padding: 20px; border-radius: 12px; text-align: center; color: white; }}
        .bg-green {{ background: #51cf66; }}
        .bg-red {{ background: #ff6b6b; }}
        .bg-grey {{ background: #adb5bd; }}
        .stat-value {{ font-size: 28px; font-weight: bold; }}
        .plotly .hoverlayer .hovertext {{ max-width: none !important; width: auto !important; white-space: normal !important; word-wrap: break-word !important; }}
        #chart-area {{ width: 100%; height: {max(600, len(source_ids) * 50)}px; overflow: auto; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Policy Compliance Overview</h1>
            <p>Visual status of CSIRO policies vs WHS regulations</p>
        </div>
        
        <div class="stats-bar">
            <div class="stat-card bg-green">
                <div class="stat-value">{stats['compliant']}</div>
                <div>Compliant Items</div>
            </div>
            <div class="stat-card bg-red">
                <div class="stat-value">{stats['non_compliant']}</div>
                <div>Non-Compliant Items</div>
            </div>
        </div>

        <div id="chart-area"></div>
    </div>

    <script>
        const hoverTexts = {json.dumps(scatter_text)};
        const hoverInfoArray = hoverTexts.map(t => t ? 'text' : 'skip');
        const data = [{{
            x: {json.dumps(scatter_x)},
            y: {json.dumps(scatter_y)},
            mode: 'markers',
            marker: {{
                size: 30,
                symbol: 'square',
                color: {json.dumps(scatter_colors)},
                line: {{ color: 'white', width: 2 }}
            }},
            text: {json.dumps(scatter_text)},
            hoverinfo: 'text',
            type: 'scatter'
        }}];

        const layout = {{
            showlegend: false,
            plot_bgcolor: '#ffffff',
            xaxis: {{
                title: {{ 
                    text: '<b>Model WHS regulation</b>', 
                    font: {{ size: 14, color: '#666' }},
                    standoff: 20
                }},
                showgrid: false,
                zeroline: false,
                tickvals: {json.dumps(list(range(len(provision_ids))))},
                ticktext: {json.dumps(provision_ids)},
                tickfont: {{ size: 12, color: '#333' }},
                side: 'top',
                tickangle: 0,
                tickposition: 'inside'
            }},
            yaxis: {{
                title: {{ 
                    text: '<b>CSIRO Electrical policies</b>', 
                    font: {{ size: 14, color: '#666' }},
                    standoff: 20
                }},
                showgrid: false,
                zeroline: false,
                tickvals: {json.dumps(list(range(len(source_ids))))},
                ticktext: {json.dumps(source_ids[::-1])}, 
                tickfont: {{ size: 12, color: '#333' }},
                tickposition: 'inside'
            }},
            margin: {{ l: 180, r: 50, t: 120, b: 50 }},
            hovermode: 'closest'
        }};

        Plotly.newPlot('chart-area', data, layout, {{responsive: true, displayModeBar: false}});
    </script>
</body>
</html>"""
    return html

if __name__ == "__main__":
    print("=" * 70)
    print("Generating Interactive Compliance Heatmap")
    print("=" * 70)
    create_interactive_heatmap()
    print("\n" + "=" * 70)
    print("✓ Interactive heatmap generated successfully!")
    print("✓ Open: compliance_check/result/compliance_heatmap.html in your browser")
    print("=" * 70)
