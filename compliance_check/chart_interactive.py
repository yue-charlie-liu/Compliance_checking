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

    print(f"Found {len(source_titles)} relevant source titles")
    print(f"Found {len(provision_titles)} relevant provision titles")
    print(f"Creating interactive {len(source_titles)} x {len(provision_titles)} heatmap")

    heatmap_data = np.zeros((len(source_titles), len(provision_titles)), dtype=int)
    hover_texts = [["" for _ in range(len(provision_titles))] for _ in range(len(source_titles))]
    stats = {"compliant": 0, "non_compliant": 0, "not_relevant": 0}

    source_idx = 0
    for item in check_results:
        source_title = item.get("source_title", "").strip()
        if not source_title:
            continue
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
            hover_lines = [
                f"<b>Source ID:</b> {source_ids[source_idx]}",
                f"<b>Provision ID:</b> {provision_ids[target_idx]}",
                f"<b>Status:</b> {status}",
                f"<b>Relevance:</b> {'Yes' if is_relevant else 'No'}",
                f"<b>Match Score:</b> {match_score:.3f}",
            ]
            if is_relevant is True:
                hover_lines.append(f"<b>Compliant:</b> {'Yes' if compliant else 'No'}")
            if reasoning:
                hover_lines.append(f"<br><b>Reasoning:</b> {reasoning[:250]}")
            if remediation:
                hover_lines.append(f"<br><b>Remediation:</b> {remediation[:250]}")

            hover_texts[source_idx][target_idx] = "<br>".join(hover_lines)
        source_idx += 1

    print(f"\nCompliance Statistics:")
    print(f"  - Compliant (Green): {stats['compliant']}")
    print(f"  - Non-compliant (Red): {stats['non_compliant']}")
    print(f"  - Not Relevant (Grey): {stats['not_relevant']}")

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


def generate_html(
    heatmap_data,
    source_ids,
    provision_ids,
    hover_texts,
    stats,
    source_count,
    provision_count,
) -> str:
    colorscale = [
        [0, '#d3d3d3'],
        [0.5, '#ff6b6b'],
        [1, '#51cf66'],
    ]

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Compliance Status Interactive Heatmap</title>
    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; border-radius: 10px; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3); padding: 30px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .header h1 {{ color: #333; font-size: 28px; margin-bottom: 10px; }}
        .header p {{ color: #666; font-size: 14px; }}
        .stats {{ display: flex; justify-content: center; gap: 30px; margin-bottom: 30px; flex-wrap: wrap; }}
        .stat {{ text-align: center; padding: 15px 25px; border-radius: 8px; background: #f5f5f5; }}
        .stat.compliant {{ border-left: 4px solid #51cf66; }}
        .stat.non-compliant {{ border-left: 4px solid #ff6b6b; }}
        .stat.not-relevant {{ border-left: 4px solid #d3d3d3; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ font-size: 12px; color: #666; margin-top: 5px; }}
        .heatmap-container {{ background: #fafafa; border-radius: 8px; padding: 15px; margin-bottom: 30px; box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06); overflow-x: auto; }}
        .legend {{ display: flex; justify-content: center; gap: 30px; margin-top: 20px; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 10px; font-size: 14px; }}
        .legend-color {{ width: 30px; height: 30px; border-radius: 4px; border: 1px solid #ddd; }}
        .legend-color.compliant {{ background: #51cf66; }}
        .legend-color.non-compliant {{ background: #ff6b6b; }}
        .legend-color.not-relevant {{ background: #d3d3d3; }}
        .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 20px; padding-top: 20px; border-top: 1px solid #eee; }}
        #heatmap {{ width: 100%; min-width: 1200px; height: 700px; }}
        .info-box {{ background: #f0f7ff; border-left: 4px solid #667eea; padding: 15px; border-radius: 4px; margin-bottom: 20px; font-size: 13px; color: #333; }}
        .info-box strong {{ color: #667eea; }}
    </style>
</head>
<body>
    <div class=\"container\">
        <div class=\"header\">
            <h1>🔍 Compliance Status Interactive Heatmap</h1>
            <p>Hover over cells to view detailed compliance information.</p>
        </div>
        <div class=\"info-box\">
            <strong>ℹ️ How to use:</strong> Hover over any colored cell to see the source ID, provision ID, compliance status, match score, reasoning, and remediation.
        </div>
        <div class=\"stats\">
            <div class=\"stat compliant\"><div class=\"stat-value\">{stats['compliant']}</div><div class=\"stat-label\">Compliant</div></div>
            <div class=\"stat non-compliant\"><div class=\"stat-value\">{stats['non_compliant']}</div><div class=\"stat-label\">Non-Compliant</div></div>
            <div class=\"stat not-relevant\"><div class=\"stat-value\">{stats['not_relevant']}</div><div class=\"stat-label\">Not Relevant</div></div>
        </div>
        <div class=\"legend\">
            <div class=\"legend-item\"><div class=\"legend-color compliant\"></div><span><b>Compliant:</b> Relevant & Compliant</span></div>
            <div class=\"legend-item\"><div class=\"legend-color non-compliant\"></div><span><b>Non-Compliant:</b> Relevant & Not Compliant</span></div>
            <div class=\"legend-item\"><div class=\"legend-color not-relevant\"></div><span><b>Not Relevant:</b> Not applicable</span></div>
        </div>
        <div class=\"heatmap-container\"><div id=\"heatmap\"></div></div>
        <div class=\"footer\"><p>Generated by Compliance Checking System | {source_count} Sources × {provision_count} Provisions = {source_count * provision_count:,} cells</p></div>
    </div>
    <script>
        const heatmapData = {json.dumps(heatmap_data)};
        const sourceIds = {json.dumps(source_ids)};
        const provisionIds = {json.dumps(provision_ids)};
        const hoverTexts = {json.dumps(hover_texts)};
        const colorscale = {json.dumps(colorscale)};
        const trace = {{
            z: heatmapData,
            x: provisionIds,
            y: sourceIds,
            type: 'heatmap',
            colorscale: colorscale,
            customdata: hoverTexts,
            hovertemplate: '%{{customdata}}<extra></extra>',
            colorbar: {{
                title: 'Status',
                tickvals: [0, 1, 2],
                ticktext: ['Not Relevant', 'Non-Compliant', 'Compliant'],
                thickness: 20,
                len: 0.7
            }},
            hoverinfo: 'skip'
        }};
        const minCellSize = 24;
        const maxWidth = 3200;
        const maxHeight = 2400;
        const width = Math.min(maxWidth, Math.max(1200, provisionIds.length * minCellSize));
        const height = Math.min(maxHeight, Math.max(700, sourceIds.length * minCellSize));
        const layout = {{
            title: {{ text: '<b>Compliance Status Heatmap</b><br><sub>Hover over cells to view details</sub>', x: 0.5, xanchor: 'center', font: {{ size: 18 }} }},
            xaxis: {{ title: 'WHS Regulation Provisions (' + provisionIds.length + ' items)', showticklabels: false }},
            yaxis: {{ title: 'Source Documents (' + sourceIds.length + ' items)', showticklabels: false }},
            width: width,
            height: height,
            margin: {{ l: 120, r: 200, t: 120, b: 120 }},
            plot_bgcolor: '#fff',
            paper_bgcolor: '#fff',
            font: {{ family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif' }}
        }};
        const config = {{ responsive: true, displayModeBar: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d'] }};
        Plotly.newPlot('heatmap', [trace], layout, config);
        window.addEventListener('resize', () => Plotly.Plots.resize('heatmap'));
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
