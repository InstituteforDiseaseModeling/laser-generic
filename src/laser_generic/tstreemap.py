"""
D3.js HTML Treemap Generator for TimingStats

This module provides functionality to generate interactive HTML treemaps
using D3.js to visualize hierarchical timing data from TimingStats objects.
"""

import json
from pathlib import Path
from typing import Any


def _convert_to_d3_hierarchy(timing_data: dict[str, dict[str, Any]], scale_factor: int = 1_000_000) -> dict[str, Any]:
    """
    Convert TimingStats data to D3.js hierarchical format.

    Args:
        timing_data: The _timing_data from a TimingStats object
        scale_factor: Factor to convert nanoseconds to desired unit (default: ms)

    Returns:
        Dictionary in D3 hierarchy format with name, value, and children
    """

    def build_node(label: str) -> dict[str, Any]:
        data = timing_data[label]
        node = {
            "name": label,
            "value": data["total_time"] / scale_factor,
            "self_value": data["self_time"] / scale_factor,
            "call_count": data["call_count"],
            "execution_order": data["execution_order"],
        }

        if data["children"]:
            # Sort children by execution order
            sorted_children = sorted(data["children"], key=lambda child: timing_data[child]["execution_order"])
            node["children"] = [build_node(child) for child in sorted_children if child in timing_data]

        return node

    # Find root nodes (those without parents)
    root_nodes = [label for label, data in timing_data.items() if data["parent"] is None and label != "__global__"]

    if not root_nodes:
        # If no explicit root nodes, create a wrapper with global data
        global_data = timing_data.get("__global__", {})
        return {
            "name": "Total Execution",
            "value": global_data.get("total_time", 0) / scale_factor,
            "self_value": global_data.get("self_time", 0) / scale_factor,
            "call_count": global_data.get("call_count", 1),
            "execution_order": 0,
            "children": [],
        }

    # Sort root nodes by execution order
    root_nodes.sort(key=lambda label: timing_data[label]["execution_order"])

    if len(root_nodes) == 1:
        return build_node(root_nodes[0])
    else:
        # Multiple root nodes - create wrapper
        total_time = sum(timing_data[root]["total_time"] for root in root_nodes)
        return {
            "name": "Total Execution",
            "value": total_time / scale_factor,
            "self_value": 0,
            "call_count": 1,
            "execution_order": 0,
            "children": [build_node(root) for root in root_nodes],
        }


def generate_d3_treemap_html(
    timing_stats, output_file: str, title: str = "Timing Analysis Treemap", scale: str = "ms", width: int = 1200, height: int = 800
) -> None:
    """
    Generate an interactive HTML file with D3.js treemap visualization.

    Args:
        timing_stats: TimingStats object (must be frozen)
        output_file: Path to output HTML file
        title: Title for the visualization
        scale: Time scale ('ns', '�s', 'ms', 's')
        width: Width of the treemap in pixels
        height: Height of the treemap in pixels
    """
    if not timing_stats._frozen:
        raise RuntimeError("TimingStats must be frozen before generating treemap")

    scale_factors = {
        "ns": (1, "ns"),
        "microseconds": (1_000, "µs"),
        "µs": (1_000, "µs"),
        "milliseconds": (1_000_000, "ms"),
        "ms": (1_000_000, "ms"),
        "seconds": (1_000_000_000, "s"),
        "s": (1_000_000_000, "s"),
    }

    if scale not in scale_factors:
        raise ValueError(f"Invalid scale '{scale}'. Valid options: {list(scale_factors.keys())}")

    scale_factor, scale_unit = scale_factors[scale]

    # Convert timing data to D3 format
    hierarchy_data = _convert_to_d3_hierarchy(timing_stats._timing_data, scale_factor)

    # Generate HTML content
    html_content = _generate_html_template(hierarchy_data, title, scale_unit, width, height)

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")


def _generate_html_template(data: dict[str, Any], title: str, scale_unit: str, width: int, height: int) -> str:
    """Generate the complete HTML template with embedded D3.js treemap."""

    data_json = json.dumps(data, indent=2)

    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}

        .container {{
            max-width: {width + 40}px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        h1 {{
            text-align: center;
            color: #2c3e50;
            margin-bottom: 10px;
        }}

        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
            font-size: 14px;
        }}

        .treemap-container {{
            text-align: center;
            margin-bottom: 20px;
        }}

        .node {{
            stroke: #fff;
            stroke-width: 1px;
            cursor: pointer;
            transition: opacity 0.3s;
        }}

        .node:hover {{
            opacity: 0.8;
        }}

        .node-text {{
            fill: #2c3e50;
            font-size: 12px;
            font-weight: bold;
            text-anchor: middle;
            pointer-events: none;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.8);
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}

        .legend {{
            margin-top: 20px;
            text-align: center;
        }}

        .legend-item {{
            display: inline-block;
            margin: 0 10px;
            font-size: 12px;
            color: #666;
        }}

        .color-box {{
            display: inline-block;
            width: 15px;
            height: 15px;
            margin-right: 5px;
            vertical-align: middle;
            border: 1px solid #ddd;
        }}

        .stats {{
            margin-top: 20px;
            text-align: center;
            font-size: 14px;
            color: #666;
        }}

        .breadcrumb {{
            text-align: center;
            margin-bottom: 10px;
            font-size: 14px;
            color: #666;
        }}

        .breadcrumb a {{
            color: #3498db;
            text-decoration: none;
            cursor: pointer;
        }}

        .breadcrumb a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="subtitle">Interactive treemap showing execution time hierarchy</div>
        <div class="breadcrumb" id="breadcrumb"></div>
        <div class="treemap-container">
            <svg id="treemap" width="{width}" height="{height}"></svg>
        </div>
        <div class="stats" id="stats"></div>
        <div class="legend">
            <div class="legend-item">
                <span class="color-box" style="background: #3498db;"></span>
                Depth 0 (Root)
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #e74c3c;"></span>
                Depth 1
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #2ecc71;"></span>
                Depth 2
            </div>
            <div class="legend-item">
                <span class="color-box" style="background: #f39c12;"></span>
                Depth 3+
            </div>
        </div>
    </div>

    <div class="tooltip" id="tooltip" style="display: none;"></div>

    <script>
        const data = {data_json};
        const scaleUnit = "{scale_unit}";
        const width = {width};
        const height = {height};

        // Color scheme for different depths
        const colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
                       '#1abc9c', '#34495e', '#f1c40f', '#e67e22', '#95a5a6'];

        let currentData = data;
        let breadcrumbs = [];

        const svg = d3.select("#treemap");
        const tooltip = d3.select("#tooltip");
        const breadcrumb = d3.select("#breadcrumb");
        const stats = d3.select("#stats");

        function getColor(d) {{
            return colors[d.depth % colors.length];
        }}

        function formatValue(value) {{
            return value.toFixed(2) + scaleUnit;
        }}

        function updateBreadcrumb() {{
            if (breadcrumbs.length === 0) {{
                breadcrumb.html("");
                return;
            }}

            let html = '<a onclick="navigateToRoot()">Root</a>';
            breadcrumbs.forEach((item, index) => {{
                html += ' � <a onclick="navigateTo(' + index + ')">' + item.name + '</a>';
            }});
            breadcrumb.html(html);
        }}

        function updateStats(data) {{
            const totalValue = data.value;
            const callCount = data.call_count;
            const nodeCount = countNodes(data);

            stats.html(`
                <strong>Total Time:</strong> ${{formatValue(totalValue)}} |
                <strong>Calls:</strong> ${{callCount}} |
                <strong>Components:</strong> ${{nodeCount}}
            `);
        }}

        function countNodes(node) {{
            let count = 1;
            if (node.children) {{
                node.children.forEach(child => {{
                    count += countNodes(child);
                }});
            }}
            return count;
        }}

        function drawTreemap(data) {{
            svg.selectAll("*").remove();

            const root = d3.hierarchy(data)
                .sum(d => d.children ? 0 : d.value)  // Only leaf nodes contribute to size
                .sort((a, b) => (a.data.execution_order || 0) - (b.data.execution_order || 0));

            d3.treemap()
                .size([width, height])
                .padding(2)
                (root);

            const leaf = svg.selectAll("g")
                .data(root.leaves())
                .join("g")
                .attr("transform", d => `translate(${{d.x0}},${{d.y0}})`);

            leaf.append("rect")
                .attr("class", "node")
                .attr("width", d => d.x1 - d.x0)
                .attr("height", d => d.y1 - d.y0)
                .attr("fill", d => getColor(d))
                .on("mouseover", function(event, d) {{
                    const selfValue = d.data.self_value || 0;
                    tooltip
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px")
                        .html(`
                            <strong>${{d.data.name}}</strong><br/>
                            Total Time: ${{formatValue(d.data.value)}}<br/>
                            Self Time: ${{formatValue(selfValue)}}<br/>
                            Calls: ${{d.data.call_count}}<br/>
                            Depth: ${{d.depth}}
                        `);
                }})
                .on("mousemove", function(event) {{
                    tooltip
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    tooltip.style("display", "none");
                }})
                .on("click", function(event, d) {{
                    // Navigate to this node if it has children
                    const originalNode = findNodeByPath(data, getNodePath(d));
                    if (originalNode && originalNode.children && originalNode.children.length > 0) {{
                        navigateToNode(originalNode);
                    }}
                }});

            leaf.append("text")
                .attr("class", "node-text")
                .attr("x", d => (d.x1 - d.x0) / 2)
                .attr("y", d => (d.y1 - d.y0) / 2)
                .attr("dy", "0.35em")
                .text(d => {{
                    const width = d.x1 - d.x0;
                    const height = d.y1 - d.y0;
                    if (width < 60 || height < 30) return "";

                    let name = d.data.name;
                    if (name.length > 12) {{
                        name = name.substring(0, 12) + "...";
                    }}
                    return name;
                }})
                .append("tspan")
                .attr("x", d => (d.x1 - d.x0) / 2)
                .attr("dy", "1.2em")
                .text(d => {{
                    const width = d.x1 - d.x0;
                    const height = d.y1 - d.y0;
                    if (width < 80 || height < 45) return "";
                    return formatValue(d.data.value);
                }});
        }}

        function getNodePath(node) {{
            const path = [];
            let current = node;
            while (current && current.parent) {{
                path.unshift(current.data.name);
                current = current.parent;
            }}
            return path;
        }}

        function findNodeByPath(root, path) {{
            let current = root;
            for (const segment of path) {{
                if (!current.children) return null;
                current = current.children.find(child => child.name === segment);
                if (!current) return null;
            }}
            return current;
        }}

        function navigateToNode(nodeData) {{
            breadcrumbs.push({{name: nodeData.name, data: nodeData}});
            currentData = nodeData;
            updateBreadcrumb();
            updateStats(nodeData);
            drawTreemap(nodeData);
        }}

        function navigateTo(index) {{
            if (index < 0) {{
                navigateToRoot();
                return;
            }}

            breadcrumbs = breadcrumbs.slice(0, index + 1);
            currentData = breadcrumbs[index].data;
            updateBreadcrumb();
            updateStats(currentData);
            drawTreemap(currentData);
        }}

        function navigateToRoot() {{
            breadcrumbs = [];
            currentData = data;
            updateBreadcrumb();
            updateStats(data);
            drawTreemap(data);
        }}

        // Initial render
        updateStats(data);
        drawTreemap(data);
    </script>
</body>
</html>'''

    return html_template
