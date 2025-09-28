"""
D3.js HTML Treemap Generator for TimingStats

This module provides functionality to generate interactive HTML treemaps
using D3.js to visualize hierarchical timing data from TimingStats objects.
"""

import json
from pathlib import Path
from typing import Any

from laser_generic.newutils import TimingStats


def _convert_to_d3_hierarchy(timing_stats: TimingStats, scale_factor: int = 1_000_000) -> dict[str, Any]:
    """
    Convert TimingStats data to D3.js hierarchical format.

    Args:
        timing_data: The _timing_data from a TimingStats object
        scale_factor: Factor to convert nanoseconds to desired unit (default: ms)

    Returns:
        Dictionary in D3 hierarchy format with name, value, and children
    """

    def build_node(tc: Any, depth: int) -> dict[str, Any]:
        total_time = tc.elapsed / scale_factor
        mean_time = tc.elapsed / tc.ncalls / scale_factor
        exclusive_time = tc.exclusive / scale_factor
        node = {
            "name": tc.label,
            "value": total_time,
            "mean": mean_time,
            "exclusive": exclusive_time,
            "call_count": tc.ncalls,
            "depth": depth,
        }

        if tc.children:
            node["children"] = [build_node(child, depth + 1) for child in tc.children.values()]

        return node

    root = build_node(timing_stats.root, 1)

    return root


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
    if not timing_stats.frozen:
        raise RuntimeError("TimingStats must be frozen before generating treemap")

    scale_factors = {
        "ns": (1, "ns"),
        "nanoseconds": (1, "ns"),
        "us": (1_000, "µs"),
        "µs": (1_000, "µs"),
        "microseconds": (1_000, "µs"),
        "ms": (1_000_000, "ms"),
        "milliseconds": (1_000_000, "ms"),
        "s": (1_000_000_000, "s"),
        "seconds": (1_000_000_000, "s"),
    }

    if scale not in scale_factors:
        raise ValueError(f"Invalid scale '{scale}'. Valid options: {list(scale_factors.keys())}")

    scale_factor, scale_unit = scale_factors[scale]

    # Convert timing data to D3 format
    hierarchy_data = _convert_to_d3_hierarchy(timing_stats, scale_factor)

    # Generate HTML content
    html_content = _generate_html_template(hierarchy_data, title, scale_unit)

    # Write to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")

    return


def _generate_html_template(data: dict[str, Any], title: str, scale_unit: str) -> str:
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
            width: 100vw;
            height: 100vh;
            margin: 0;
            padding: 10px;
            box-sizing: border-box;
            background: white;
            display: flex;
            flex-direction: column;
        }}

        h1 {{
            text-align: center;
            color: #2c3e50;
            margin: 0 0 5px 0;
            font-size: 1.5em;
        }}

        .subtitle {{
            text-align: center;
            color: #7f8c8d;
            margin: 0 0 10px 0;
            font-size: 14px;
        }}

        .treemap-container {{
            flex: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 10px 0;
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
            fill: black;
            font-size: 12px;
            font-weight: bold;
            text-anchor: middle;
            pointer-events: none;
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
            <svg id="treemap"></svg>
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
        let width, height;

        function updateDimensions() {{
            const container = document.querySelector('.treemap-container');
            const rect = container.getBoundingClientRect();
            width = Math.max(400, rect.width - 20);
            height = Math.max(300, rect.height - 20);

            d3.select("#treemap")
                .attr("width", width)
                .attr("height", height);
        }}

        let currentData = data;
        let breadcrumbs = [];

        const svg = d3.select("#treemap");
        const tooltip = d3.select("#tooltip");
        const breadcrumb = d3.select("#breadcrumb");
        const stats = d3.select("#stats");

        // Depth-based color scale for nested treemap
        const colorScale = d3.scaleSequential([8, 0], d3.interpolateMagma);

        function getColor(d) {{
            return colorScale(d.depth);
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
                html += ' > <a onclick="navigateTo(' + index + ')">' + item.name + '</a>';
            }});
            breadcrumb.html(html);
        }}

        function updateStats(data) {{
            const totalValue = data.value; // total time
            const callCount = data.call_count;  // total calls
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
                .sum(d => d.value || 0);
                // .sort((a, b) => (a.data.execution_order || 0) - (b.data.execution_order || 0));

            d3.treemap()
                .size([width, height])
                .paddingOuter(3)
                .paddingTop(19)
                .paddingInner(1)
                (root);

            // Add drop shadow filter for depth perception
            const defs = svg.append("defs");
            const filter = defs.append("filter")
                .attr("id", "drop-shadow")
                .attr("height", "130%");

            filter.append("feGaussianBlur")
                .attr("in", "SourceAlpha")
                .attr("stdDeviation", 2);

            filter.append("feOffset")
                .attr("dx", 2)
                .attr("dy", 2)
                .attr("result", "offset");

            const feMerge = filter.append("feMerge");
            feMerge.append("feMergeNode")
                .attr("in", "offset");
            feMerge.append("feMergeNode")
                .attr("in", "SourceGraphic");

            // Render all nodes including parents to show nesting
            const node = svg.selectAll("g")
                .data(root.descendants())  // Show all nodes including parents
                .join("g")
                .attr("transform", d => `translate(${{d.x0}},${{d.y0}})`);

            node.append("rect")
                .attr("class", "node")
                .attr("width", d => d.x1 - d.x0)
                .attr("height", d => d.y1 - d.y0)
                .attr("fill", d => getColor(d))
                .attr("fill-opacity", d => d.children ? 0.6 : 1)
                .style("filter", "url(#drop-shadow)")
                .on("mouseover", function(event, d) {{
                    const selfValue = d.data.exclusive || 0;
                    tooltip
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px")
                        .html(`
                            <strong>${{d.data.name}}</strong><br/>
                            Total Time: ${{formatValue(d.data.value)}}<br/>
                            Self Time: ${{formatValue(selfValue)}}<br/>
                            Calls: ${{d.data.call_count}}<br/>
                            Mean Time: ${{formatValue(d.data.mean)}}<br/>
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

            // Add labels for parent nodes (in top padding area)
            node.filter(d => d.children)
                .append("text")
                .attr("class", "node-text")
                .attr("x", 4)
                .attr("y", 13)
                .style("font-size", "11px")
                .style("font-weight", "bold")
                .style("fill", "black")
                .style("text-anchor", "start")
                .text(d => d.data.name);

            // Add labels for leaf nodes (centered)
            node.filter(d => !d.children)
                .append("text")
                .attr("class", "node-text")
                .attr("x", d => (d.x1 - d.x0) / 2)
                .attr("y", d => (d.y1 - d.y0) / 2)
                .attr("dy", "0.35em")
                .style("text-anchor", "middle")
                .style("fill", "black")
                .style("font-weight", "bold")
                .text(d => {{
                    const width = d.x1 - d.x0;
                    const height = d.y1 - d.y0;
                    if (width < 60 || height < 30) return "";
                    return d.data.name;
                }})
                .append("tspan")
                .attr("x", d => (d.x1 - d.x0) / 2)
                .attr("dy", "1.2em")
                .style("fill-opacity", 0.8)
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

        // Handle window resize
        window.addEventListener('resize', function() {{
            updateDimensions();
            drawTreemap(currentData);
        }});

        // Initial render
        updateDimensions();
        updateStats(data);
        drawTreemap(data);
    </script>
</body>
</html>'''

    return html_template
