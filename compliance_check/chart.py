import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
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
    
    return id_part.strip() or title[:20]  # Return ID or first 20 chars if no ID found


def create_compliance_heatmap():
    """
    Create a heatmap of compliance results.
    
    Heatmap structure:
    - Y-axis: Source titles
    - X-axis: WHS regulation provision titles
    - Colors:
        - Green: relevant AND compliant
        - Red: relevant AND NOT compliant
        - Grey: NOT relevant or null
    """
    
    # Define file paths
    check_results_file = Path(__file__).resolve().parent / "result" / "check_results.json"
    whs_regulations_file = Path(__file__).resolve().parent.parent / "chunking" / "flattened" / "model WHS regulations.json"
    output_file = Path(__file__).resolve().parent / "result" / "compliance_heatmap.png"
    stats_file = Path(__file__).resolve().parent / "result" / "compliance_stats.json"
    
    # Load data
    print(f"Loading check results from {check_results_file}")
    check_results = load_json(check_results_file)
    
    print(f"Loading WHS regulations from {whs_regulations_file}")
    whs_regulations = load_json(whs_regulations_file)
    
    # Extract provision titles from WHS regulations (filter out empty titles)
    provision_titles = []
    provision_ids = []
    for item in whs_regulations:
        title = item.get("title", "").strip()
        if title:
            provision_titles.append(title)
            provision_ids.append(extract_id(title))
    
    print(f"Found {len(provision_titles)} provision titles")
    
    # Extract source titles
    source_titles = []
    source_ids = []
    for item in check_results:
        title = item.get("source_title", "").strip()
        if title:
            source_titles.append(title)
            source_ids.append(extract_id(title))
    
    print(f"Found {len(source_titles)} source titles")
    print(f"Creating {len(source_titles)} x {len(provision_titles)} heatmap")
    
    # Create a matrix for the heatmap
    # Value mapping:
    # 0 = grey (not relevant or null compliance)
    # 1 = red (relevant but NOT compliant)
    # 2 = green (relevant AND compliant)
    heatmap_data = np.zeros((len(source_titles), len(provision_titles)))
    
    # Track statistics
    stats = {
        "total_cells": len(source_titles) * len(provision_titles),
        "compliant": 0,
        "non_compliant": 0,
        "not_relevant": 0,
        "null_compliance": 0
    }
    
    # Fill the heatmap
    for source_idx, item in enumerate(check_results):
        source_title = item.get("source_title", "").strip()
        matches = item.get("matches", [])
        
        for match in matches:
            target_title = match.get("target_title", "").strip()
            compliance = match.get("compliance", {})
            
            # Find the column index for this target title
            if target_title not in provision_titles:
                continue
            
            target_idx = provision_titles.index(target_title)
            
            is_relevant = compliance.get("is_relevant", None)
            compliant = compliance.get("compliant", None)
            
            if is_relevant is True and compliant is True:
                heatmap_data[source_idx, target_idx] = 2  # Green
                stats["compliant"] += 1
            elif is_relevant is True and compliant is False:
                heatmap_data[source_idx, target_idx] = 1  # Red
                stats["non_compliant"] += 1
            elif is_relevant is False:
                heatmap_data[source_idx, target_idx] = 0  # Grey
                stats["not_relevant"] += 1
            else:
                heatmap_data[source_idx, target_idx] = 0  # Grey
                stats["null_compliance"] += 1
    
    # Calculate percentages
    total_colored = stats["compliant"] + stats["non_compliant"] + stats["not_relevant"]
    if total_colored > 0:
        stats["compliance_rate"] = round(stats["compliant"] / (stats["compliant"] + stats["non_compliant"]) * 100, 2) if (stats["compliant"] + stats["non_compliant"]) > 0 else 0
        stats["relevance_rate"] = round((stats["compliant"] + stats["non_compliant"]) / total_colored * 100, 2)
    
    print(f"\nCompliance Statistics:")
    print(f"  - Compliant (Green): {stats['compliant']}")
    print(f"  - Non-compliant (Red): {stats['non_compliant']}")
    print(f"  - Not Relevant (Grey): {stats['not_relevant']}")
    print(f"  - Null Compliance: {stats['null_compliance']}")
    print(f"  - Compliance Rate: {stats.get('compliance_rate', 0)}%")
    print(f"  - Relevance Rate: {stats.get('relevance_rate', 0)}%")
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(32, 20))
    
    # Define colors: grey, red, green
    cmap_colors = ['#d3d3d3', '#ff6b6b', '#51cf66']  # Grey, Red, Green
    cmap = sns.color_palette(cmap_colors)
    
    # Create heatmap without tick labels (too many to display)
    sns.heatmap(
        heatmap_data,
        cmap=cmap,
        cbar=False,
        ax=ax,
        xticklabels=False,  # Hide x-axis labels (too many)
        yticklabels=False,  # Hide y-axis labels (too many)
        linewidths=0.1,
        linecolor='white',
        vmin=0,
        vmax=2,
        square=False
    )
    
    # Customize plot
    ax.set_title(
        f'Compliance Status Heatmap\n({len(source_ids)} Sources × {len(provision_ids)} Provisions)\nCompliant: {stats["compliant"]}, Non-compliant: {stats["non_compliant"]}, Not Relevant: {stats["not_relevant"]}',
        fontsize=18,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel(f'WHS Regulation Provisions ({len(provision_ids)} items)', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Source Documents ({len(source_ids)} items)', fontsize=14, fontweight='bold')
    
    # Add legend
    green_patch = mpatches.Patch(color='#51cf66', label='Compliant (Relevant & Compliant)')
    red_patch = mpatches.Patch(color='#ff6b6b', label='Non-Compliant (Relevant & Not Compliant)')
    grey_patch = mpatches.Patch(color='#d3d3d3', label='Not Relevant')
    ax.legend(
        handles=[green_patch, red_patch, grey_patch],
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10
    )
    
    plt.tight_layout()
    
    # Save figure
    print(f"\nSaving heatmap to {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved to {output_file}")
    
    # Save statistics
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ Statistics saved to {stats_file}")
    
    plt.close()


def create_summary_stats():
    """Create additional summary statistics."""
    check_results_file = Path(__file__).resolve().parent / "result" / "check_results.json"
    summary_file = Path(__file__).resolve().parent / "result" / "compliance_summary.json"
    
    check_results = load_json(check_results_file)
    
    summary = {
        "total_sources": len(check_results),
        "source_summaries": []
    }
    
    for item in check_results:
        source_title = item.get("source_title", "")
        matches = item.get("matches", [])
        
        compliant_count = 0
        non_compliant_count = 0
        not_relevant_count = 0
        
        for match in matches:
            compliance = match.get("compliance", {})
            is_relevant = compliance.get("is_relevant", None)
            compliant = compliance.get("compliant", None)
            
            if is_relevant is True and compliant is True:
                compliant_count += 1
            elif is_relevant is True and compliant is False:
                non_compliant_count += 1
            else:
                not_relevant_count += 1
        
        total_relevant = compliant_count + non_compliant_count
        compliance_rate = round(compliant_count / total_relevant * 100, 2) if total_relevant > 0 else None
        
        summary["source_summaries"].append({
            "source_title": source_title,
            "total_matches": len(matches),
            "compliant": compliant_count,
            "non_compliant": non_compliant_count,
            "not_relevant": not_relevant_count,
            "compliance_rate": compliance_rate
        })
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Summary saved to {summary_file}")


if __name__ == "__main__":
    print("=" * 70)
    print("Generating Compliance Visualizations")
    print("=" * 70)
    
    create_compliance_heatmap()
    create_summary_stats()
    
    print("\n" + "=" * 70)
    print("✓ All charts generated successfully!")
    print("=" * 70)
