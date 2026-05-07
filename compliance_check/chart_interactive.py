import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
import warnings

warnings.filterwarnings('ignore')


def load_json(path: Path) -> Any:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_id(title: str) -> str:
    """
    Extract ID from title.
    ID format: combination of numbers, dots, and letters at the beginning.
    Example: "1.1 Introduction" -> "1.1"
             "A.1.2 Safety" -> "A.1.2"
    """
    if not title:
        return ""
    
    id_part = ""
    for char in title:
        if char.isdigit() or char in '.-' or char.isalpha():
            id_part += char
        elif char == ' ':
            # Stop at first space after ID
            break
        else:
            break
    
    return id_part.strip() or title[:20]


def create_interactive_heatmap():
    """
    Create an interactive HTML heatmap of compliance results using Plotly.
    """
    
    # Define file paths
    check_results_file = Path(__file__).resolve().parent / "result" / "check_results.json"
    whs_regulations_file = Path(__file__).resolve().parent.parent / "chunking" / "flattened" / "model WHS regulations.json"
    output_file = Path(__file__).resolve().parent / "result" / "compliance_heatmap.html"
    
    # Load data
    print(f"Loading check results from {check_results_file}")
    check_results = load_json(check_results_file)
    
    print(f"Loading WHS regulations from {whs_regulations_file}")
    whs_regulations = load_json(whs_regulations_file)
    
    # Extract provision titles and IDs
    provision_titles = []
    provision_ids = []
    provision_map = {}  # Map title to full info
    for item in whs_regulations:
        title = item.get("title", "").strip()
        if title:
            provision_titles.append(title)
            provision_ids.append(extract_id(title))
            provision_map[title] = item
    
    print(f"Found {len(provision_titles)} provision titles")
    
    # Extract source titles and IDs
    source_titles = []
    source_ids = []
    source_map = {}  # Map title to full info
    for item in check_results:
        title = item.get("source_title", "").strip()
        if title:
            source_titles.append(title)
            source_ids.append(extract_id(title))
            source_map[title] = item
    
    print(f"Found {len(source_titles)} source titles")
    print(f"Creating interactive {len(source_titles)} x {len(provision_titles)} heatmap")
    
    # Create data structure for heatmap
    heatmap_data = np.zeros((len(source_titles), len(provision_titles)))
    hover_texts = [['' for _ in provision_titles] for _ in source_titles]
    
    # Track statistics
    stats = {
        "compliant": 0,
        "non_compliant": 0,
        "not_relevant": 0,
    }
    
    # Fill the heatmap
    for source_idx, item in enumerate(check_results):
        source_title = item.get("source_title", "").strip()
        source_id = source_ids[source_idx]
        matches = item.get("matches", [])
        
        for match in matches:
            target_title = match.get("target_title", "").strip()
            compliance = match.get("compliance", {})
            
            # Find the column index for this target title
            if target_title not in provision_titles:
                continue
            
            target_idx = provision_titles.index(target_title)
            target_id = provision_ids[target_idx]
            
            is_relevant = compliance.get("is_relevant", None)
            compliant = compliance.get("compliant", None)
            reasoning = compliance.get("reasoning", "")
            remediation = compliance.get("remediation", "")
            match_score = match.get("match_score", 0.0)
            
            status = "Not Relevant"
            color_value = 0
            
            if is_relevant is True and compliant is True:
                status = "✓ COMPLIANT"
                color_value = 2
                stats["compliant"] += 1
            elif is_relevant is True and compliant is False:
                status = "✗ NON-COMPLIANT"
                color_value = 1
                stats["non_compliant"] += 1
            else:
                status = "Not Relevant"
                color_value = 0
                stats["not_relevant"] += 1
            
            heatmap_data[source_idx, target_idx] = color_value
            
            # Create hover text with formatted information
            hover_text = f"<b>Source:</b> {source_id}<br>"
            hover_text += f"<b>Provision:</b> {target_id}<br>"
            hover_text += f"<b>Status:</b> {status}<br>"
            hover_text += f"<b>Relevance:</b> {'Yes' if is_relevant else 'No'}<br>"
            if is_relevant:
                hover_text += f"<b>Compliant:</b> {'Yes' if compliant else 'No'}<br>"
            hover_text += f"<b>Match Score:</b> {match_score:.3f}<br>"
            if reasoning:
                hover_text += f"<br><b>Reasoning:</b><br>{reasoning[:200]}..."
            if remediation and remediation.strip():
                hover_text += f"<br><br><b>Remediation:</b><br>{remediation[:200]}..."
            
            hover_texts[source_idx][target_idx] = hover_text
    
    print(f"\nCompliance Statistics:")
    print(f"  - Compliant (Green): {stats['compliant']}")
    print(f"  - Non-compliant (Red): {stats['non_compliant']}")
    print(f"  - Not Relevant (Grey): {stats['not_relevant']}")
    
    # Generate HTML
    html_content = generate_html(
        heatmap_data, 
        source_ids, 
        provision_ids,
        hover_texts,
        stats
    )
    
    # Save HTML file
    print(f"\nSaving interactive heatmap to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"✓ Interactive heatmap saved to {output_file}")


def generate_html(heatmap_data, source_ids, provision_ids, hover_texts, stats):
    """Generate HTML with interactive heatmap using Plotly."""
    
    # Create colorscale: [0: grey, 1: red, 2: green]
    colorscale = [
        [0, '#d3d3d3'],      # Grey for not relevant
        [0.5, '#ff6b6b'],    # Red for non-compliant
        [1, '#51cf66']       # Green for compliant
    ]
    
    # Prepare hover text for Plotly
    customdata = hover_texts
    hovertemplate = "%{customdata}<extra></extra>"
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Compliance Status Interactive Heatmap</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            padding: 30px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #333;
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 14px;
        }}
        
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }}
        
        .stat {{
            text-align: center;
            padding: 15px 25px;
            border-radius: 8px;
            background: #f5f5f5;
        }}
        
        .stat.compliant {{
            border-left: 4px solid #51cf66;
        }}
        
        .stat.non-compliant {{
            border-left: 4px solid #ff6b6b;
        }}
        
        .stat.not-relevant {{
            border-left: 4px solid #d3d3d3;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        
        .heatmap-container {{
            background: #fafafa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 30px;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
        }}
        
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 14px;
        }}
        
        .legend-color {{
            width: 30px;
            height: 30px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }}
        
        .legend-color.compliant {{
            background: #51cf66;
        }}
        
        .legend-color.non-compliant {{
            background: #ff6b6b;
        }}
        
        .legend-color.not-relevant {{
            background: #d3d3d3;
        }}
        
        .footer {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #eee;
        }}
        
        #heatmap {{
            width: 100%;
            height: 600px;
        }}
        
        .info-box {{
            background: #f0f7ff;
            border-left: 4px solid #667eea;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
            font-size: 13px;
            color: #333;
        }}
        
        .info-box strong {{
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Compliance Status Interactive Heatmap</h1>
            <p>Hover over cells to see detailed compliance information</p>
        </div>
        
        <div class="info-box">
            <strong>ℹ️ How to use:</strong> Hover your mouse over any cell to view detailed compliance information. 
            Green cells indicate compliant provisions, red cells indicate non-compliant provisions, 
            and grey cells indicate provisions not relevant to the source document.
        </div>
        
        <div class="stats">
            <div class="stat compliant">
                <div class="stat-value">{stats['compliant']}</div>
                <div class="stat-label">Compliant</div>
            </div>
            <div class="stat non-compliant">
                <div class="stat-value">{stats['non_compliant']}</div>
                <div class="stat-label">Non-Compliant</div>
            </div>
            <div class="stat not-relevant">
                <div class="stat-value">{stats['not_relevant']}</div>
                <div class="stat-label">Not Relevant</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color compliant"></div>
                <span><b>Compliant:</b> Relevant & Compliant</span>
            </div>
            <div class="legend-item">
                <div class="legend-color non-compliant"></div>
                <span><b>Non-Compliant:</b> Relevant & Not Compliant</span>
            </div>
            <div class="legend-item">
                <div class="legend-color not-relevant"></div>
                <span><b>Not Relevant:</b> Not applicable to source</span>
            </div>
        </div>
        
        <div class="heatmap-container">
            <div id="heatmap"></div>
        </div>
        
        <div class="footer">
            <p>Generated by Compliance Checking System | {len(source_ids)} Sources × {len(provision_ids)} Provisions = {len(source_ids) * len(provision_ids):,} cells</p>
        </div>
    </div>
    
    <script>
        // Prepare data for Plotly
        const heatmapData = {json.dumps(heatmap_data.tolist())};
        const sourceIds = {json.dumps(source_ids)};
        const provisionIds = {json.dumps(provision_ids)};
        const hoverTexts = {json.dumps(hover_texts)};
        
        // Create the heatmap trace
        const trace = {{
            z: heatmapData,
            x: provisionIds,
            y: sourceIds,
            type: 'heatmap',
            colorscale: {json.dumps(colorscale)},
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
        
        // Define layout
        const layout = {{
            title: {{
                text: '<b>Compliance Status Heatmap</b><br><sub>Hover over cells to view details</sub>',
                x: 0.5,
                xanchor: 'center',
                font: {{ size: 18 }}
            }},
            xaxis: {{
                title: 'WHS Regulation Provisions ({len(provision_ids)} items)',
                side: 'bottom',
                showticklabels: false,
                tickangle: -45
            }},
            yaxis: {{
                title: 'Source Documents ({len(source_ids)} items)',
                showticklabels: false
            }},
            width: null,
            height: 600,
            margin: {{ l: 100, r: 200, t: 100, b: 100 }},
            plot_bgcolor: '#fff',
            paper_bgcolor: '#fff',
            font: {{ family: 'Segoe UI, Tahoma, Geneva, Verdana, sans-serif' }}
        }};
        
        // Define config
        const config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        }};
        
        // Render the plot
        Plotly.newPlot('heatmap', [trace], layout, config);
        
        // Add responsive behavior
        window.addEventListener('resize', function() {{
            Plotly.Plots.resize('heatmap');
        }});
    </script>
</body>
</html>
"""
    
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
