#!/usr/bin/env python3
"""
Generate Interactive CRP Visualization (HTML + D3.js)
======================================================

Creates an interactive web-based visualization where users can:
1. Hover/click on nodes to highlight top-K paths
2. Adjust K (number of paths to show) dynamically
3. View relevance values on edges
4. Navigate between different test images
5. Zoom and pan the graph

Author: Claude Code
Date: October 28, 2025
"""

import json
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def generate_interactive_html(crp_data_dir, output_html_path):
    """
    Generate interactive HTML visualization from CRP data

    Args:
        crp_data_dir: Path to directory containing CRP results
        output_html_path: Where to save the HTML file
    """

    crp_data_dir = Path(crp_data_dir)
    metadata_path = crp_data_dir / 'metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Extract graph data from all images
    images_data = {}

    for img_name, img_info in metadata['images'].items():
        print(f"Processing {img_name}...")

        # Load all relevance matrices for this image
        transitions = {}
        for trans_key, trans_info in img_info['transitions'].items():
            matrix_path = crp_data_dir / trans_info['file']
            rel_matrix = np.load(matrix_path)
            transitions[trans_key] = rel_matrix.tolist()

        images_data[img_name] = {
            'tile_position': img_info['tile_position'],
            'transitions': transitions
        }

    # Generate HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive U-Net CRP Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}

        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}

        .controls {{
            background: white;
            padding: 15px;
            margin: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .control-group label {{
            font-weight: 600;
            color: #333;
        }}

        select, input[type="range"] {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}

        #topK {{
            width: 200px;
        }}

        #topKValue {{
            font-weight: bold;
            color: #667eea;
            min-width: 30px;
            text-align: center;
        }}

        .visualization-container {{
            background: white;
            margin: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}

        #graph {{
            width: 100%;
            height: 800px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
        }}

        .node {{
            cursor: pointer;
            transition: all 0.2s;
        }}

        .node:hover {{
            filter: brightness(1.2);
        }}

        .node-circle {{
            stroke: white;
            stroke-width: 2px;
        }}

        .node-text {{
            font-size: 11px;
            font-weight: bold;
            pointer-events: none;
            user-select: none;
        }}

        .link {{
            fill: none;
            stroke: #999;
            stroke-opacity: 0.3;
            transition: all 0.2s;
        }}

        .link-highlighted {{
            stroke: #ff6b6b;
            stroke-width: 3px !important;
            stroke-opacity: 0.8 !important;
        }}

        .node-highlighted {{
            filter: brightness(1.3);
        }}

        .link-label {{
            font-size: 10px;
            fill: #666;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }}

        .link-label-visible {{
            opacity: 1;
        }}

        .layer-label {{
            font-size: 14px;
            font-weight: bold;
            fill: #333;
        }}

        .info-panel {{
            background: white;
            margin: 20px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .info-panel h3 {{
            margin-top: 0;
            color: #667eea;
        }}

        #selectedNodeInfo {{
            color: #666;
            line-height: 1.6;
        }}

        .legend {{
            background: white;
            margin: 20px;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .legend h3 {{
            margin-top: 0;
            color: #667eea;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}

        .legend-color {{
            width: 30px;
            height: 20px;
            margin-right: 10px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Interactive U-Net CRP Visualization</h1>
        <p>Concept Relevance Propagation: Hierarchical Channel Dependencies</p>
    </div>

    <div class="controls">
        <div class="control-group">
            <label for="imageSelect">Test Image:</label>
            <select id="imageSelect"></select>
        </div>

        <div class="control-group">
            <label for="startLayer">Start Layer:</label>
            <select id="startLayer">
                <option value="decoder_1_conv2">Decoder 1</option>
                <option value="decoder_2_conv2">Decoder 2</option>
                <option value="decoder_3_conv2">Decoder 3</option>
                <option value="decoder_4_conv2">Decoder 4</option>
            </select>
        </div>

        <div class="control-group">
            <label for="topK">Top-K Paths:</label>
            <input type="range" id="topK" min="1" max="10" value="2" step="1">
            <span id="topKValue">2</span>
        </div>

        <div class="control-group">
            <button id="resetBtn" style="padding: 8px 16px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Reset View
            </button>
        </div>
    </div>

    <div class="visualization-container">
        <svg id="graph"></svg>
    </div>

    <div class="info-panel">
        <h3>Selected Node Information</h3>
        <div id="selectedNodeInfo">Click or hover over a node to see its top contributing channels</div>
    </div>

    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item">
            <div class="legend-color" style="background: #4CAF50;"></div>
            <span>Decoder 1 (32 channels) - Final output layer</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #2196F3;"></div>
            <span>Decoder 2 (64 channels)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #9C27B0;"></div>
            <span>Decoder 3 (128 channels)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FF9800;"></div>
            <span>Decoder 4 (256 channels)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #F44336;"></div>
            <span>Bottleneck (512 channels) - Deepest layer</span>
        </div>
        <div style="margin-top: 15px; color: #666;">
            <strong>Interaction:</strong> Click or hover on a channel to highlight its top-K contributing channels from previous layers.
            Edge thickness represents relevance strength.
        </div>
    </div>

    <script>
        // Embedded data
        const metadata = {json.dumps(metadata, indent=2)};
        const imagesData = {json.dumps(images_data, indent=2)};

        // Graph configuration
        const layerSequence = metadata.layer_sequence;
        const layerChannels = metadata.layer_channels;

        // Color scheme for layers
        const layerColors = {{
            'decoder_1_conv2': '#4CAF50',
            'decoder_2_conv2': '#2196F3',
            'decoder_3_conv2': '#9C27B0',
            'decoder_4_conv2': '#FF9800',
            'bottleneck_conv2': '#F44336'
        }};

        // Initialize UI
        const imageSelect = document.getElementById('imageSelect');
        const startLayerSelect = document.getElementById('startLayer');
        const topKSlider = document.getElementById('topK');
        const topKValue = document.getElementById('topKValue');
        const resetBtn = document.getElementById('resetBtn');
        const selectedNodeInfo = document.getElementById('selectedNodeInfo');

        // Populate image selector
        Object.keys(imagesData).forEach(imgName => {{
            const option = document.createElement('option');
            option.value = imgName;
            option.textContent = imgName;
            imageSelect.appendChild(option);
        }});

        // Current state
        let currentImage = Object.keys(imagesData)[0];
        let currentStartLayer = 'decoder_1_conv2';
        let currentTopK = 2;
        let currentHighlightedNode = null;

        // Update topK display
        topKSlider.addEventListener('input', (e) => {{
            currentTopK = parseInt(e.target.value);
            topKValue.textContent = currentTopK;
            if (currentHighlightedNode) {{
                highlightTopKPaths(currentHighlightedNode);
            }}
        }});

        // Image selection change
        imageSelect.addEventListener('change', (e) => {{
            currentImage = e.target.value;
            renderGraph();
        }});

        // Start layer selection change
        startLayerSelect.addEventListener('change', (e) => {{
            currentStartLayer = e.target.value;
            renderGraph();
        }});

        // Reset button
        resetBtn.addEventListener('click', () => {{
            clearHighlights();
            currentHighlightedNode = null;
            selectedNodeInfo.innerHTML = 'Click or hover over a node to see its top contributing channels';
        }});

        // D3 setup
        const svg = d3.select('#graph');
        const width = document.getElementById('graph').clientWidth;
        const height = 800;

        svg.attr('width', width).attr('height', height);

        const g = svg.append('g');

        // Zoom behavior
        const zoom = d3.zoom()
            .scaleExtent([0.5, 3])
            .on('zoom', (event) => {{
                g.attr('transform', event.transform);
            }});

        svg.call(zoom);

        // Graph rendering function
        function renderGraph() {{
            // Clear previous graph
            g.selectAll('*').remove();
            clearHighlights();
            currentHighlightedNode = null;

            const data = imagesData[currentImage];
            const transitions = data.transitions;

            // Find start layer index
            const startLayerIndex = layerSequence.indexOf(currentStartLayer);
            const relevantLayers = layerSequence.slice(startLayerIndex);

            // Calculate layout
            const layerSpacing = width / (relevantLayers.length + 1);
            const nodes = [];
            const links = [];

            // Create nodes
            relevantLayers.forEach((layer, layerIdx) => {{
                const numChannels = layerChannels[layer];
                const channelSpacing = height / (numChannels + 1);

                for (let ch = 0; ch < numChannels; ch++) {{
                    nodes.push({{
                        id: `${{layer}}_ch${{ch}}`,
                        layer: layer,
                        channel: ch,
                        x: layerSpacing * (layerIdx + 1),
                        y: channelSpacing * (ch + 1),
                        color: layerColors[layer]
                    }});
                }}
            }});

            // Create links from relevance matrices
            for (let i = 0; i < relevantLayers.length - 1; i++) {{
                const targetLayer = relevantLayers[i];
                const sourceLayer = relevantLayers[i + 1];
                const transKey = `${{targetLayer}}_from_${{sourceLayer}}`;

                if (transitions[transKey]) {{
                    const relMatrix = transitions[transKey];

                    // For each target channel, find all source channels
                    relMatrix.forEach((sourceRelevances, targetCh) => {{
                        sourceRelevances.forEach((relevance, sourceCh) => {{
                            if (relevance > 0.001) {{  // Threshold to avoid too many links
                                const sourceNode = nodes.find(n =>
                                    n.layer === sourceLayer && n.channel === sourceCh
                                );
                                const targetNode = nodes.find(n =>
                                    n.layer === targetLayer && n.channel === targetCh
                                );

                                if (sourceNode && targetNode) {{
                                    links.push({{
                                        source: sourceNode,
                                        target: targetNode,
                                        relevance: relevance,
                                        id: `${{sourceNode.id}}_to_${{targetNode.id}}`
                                    }});
                                }}
                            }}
                        }});
                    }});
                }}
            }}

            // Draw layer labels
            g.selectAll('.layer-label')
                .data(relevantLayers)
                .enter()
                .append('text')
                .attr('class', 'layer-label')
                .attr('x', (d, i) => layerSpacing * (i + 1))
                .attr('y', 30)
                .attr('text-anchor', 'middle')
                .text(d => d.replace('_', ' ').replace('conv2', ''));

            // Draw links
            const linkGroup = g.append('g').attr('class', 'links');

            linkGroup.selectAll('.link')
                .data(links)
                .enter()
                .append('line')
                .attr('class', 'link')
                .attr('id', d => d.id)
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y)
                .attr('stroke-width', d => Math.max(0.5, d.relevance * 5));

            // Draw nodes
            const nodeGroup = g.append('g').attr('class', 'nodes');

            const node = nodeGroup.selectAll('.node')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', 'node')
                .attr('id', d => d.id)
                .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
                .on('mouseenter', function(event, d) {{
                    highlightTopKPaths(d);
                    currentHighlightedNode = d;
                }})
                .on('click', function(event, d) {{
                    highlightTopKPaths(d);
                    currentHighlightedNode = d;
                }});

            node.append('circle')
                .attr('class', 'node-circle')
                .attr('r', 6)
                .attr('fill', d => d.color);

            node.append('text')
                .attr('class', 'node-text')
                .attr('dy', '0.35em')
                .attr('text-anchor', 'middle')
                .attr('fill', 'white')
                .text(d => d.channel);
        }}

        function highlightTopKPaths(node) {{
            // Clear previous highlights
            clearHighlights();

            // Find all links where this node is the target
            const relevantLinks = [];

            // Get transition data
            const data = imagesData[currentImage];
            const transitions = data.transitions;

            // Find which layer comes before this node's layer
            const layerIdx = layerSequence.indexOf(node.layer);
            if (layerIdx < layerSequence.length - 1) {{
                const sourceLayer = layerSequence[layerIdx + 1];
                const transKey = `${{node.layer}}_from_${{sourceLayer}}`;

                if (transitions[transKey]) {{
                    const relMatrix = transitions[transKey];
                    const sourceRelevances = relMatrix[node.channel];

                    // Get top-K source channels
                    const rankedSources = sourceRelevances
                        .map((rel, idx) => ({{ channel: idx, relevance: rel }}))
                        .sort((a, b) => b.relevance - a.relevance)
                        .slice(0, currentTopK);

                    // Highlight links
                    rankedSources.forEach((src, rank) => {{
                        const linkId = `${{sourceLayer}}_ch${{src.channel}}_to_${{node.id}}`;
                        d3.select(`#${{linkId}}`)
                            .classed('link-highlighted', true);

                        // Highlight source node
                        d3.select(`#${{sourceLayer}}_ch${{src.channel}}`)
                            .classed('node-highlighted', true);
                    }});

                    // Highlight current node
                    d3.select(`#${{node.id}}`)
                        .classed('node-highlighted', true);

                    // Update info panel
                    let infoHTML = `<strong>Selected:</strong> ${{node.layer.replace('_', ' ')}} Channel ${{node.channel}}<br><br>`;
                    infoHTML += `<strong>Top ${{currentTopK}} contributing channels from ${{sourceLayer.replace('_', ' ')}}:</strong><br>`;
                    rankedSources.forEach((src, rank) => {{
                        infoHTML += `&nbsp;&nbsp;${{rank + 1}}. Channel ${{src.channel}}: <span style="color: #667eea; font-weight: bold;">${{src.relevance.toFixed(4)}}</span><br>`;
                    }});

                    selectedNodeInfo.innerHTML = infoHTML;
                }} else {{
                    selectedNodeInfo.innerHTML = `<strong>Selected:</strong> ${{node.layer.replace('_', ' ')}} Channel ${{node.channel}}<br><br>No source layer data available.`;
                }}
            }} else {{
                selectedNodeInfo.innerHTML = `<strong>Selected:</strong> ${{node.layer.replace('_', ' ')}} Channel ${{node.channel}}<br><br>This is the deepest layer (no sources).`;
            }}
        }}

        function clearHighlights() {{
            d3.selectAll('.link')
                .classed('link-highlighted', false);
            d3.selectAll('.node')
                .classed('node-highlighted', false);
        }}

        // Initial render
        renderGraph();
    </script>
</body>
</html>
"""

    # Write HTML file
    with open(output_html_path, 'w') as f:
        f.write(html_content)

    print(f"\n✓ Interactive visualization generated: {output_html_path}")
    print(f"\nTo view:")
    print(f"  1. Open {output_html_path} in a web browser")
    print(f"  2. Select a test image from the dropdown")
    print(f"  3. Click or hover on any channel to see its top-K contributors")
    print(f"  4. Adjust the 'Top-K Paths' slider to change how many paths are shown")

def main():
    parser = argparse.ArgumentParser(description='Generate Interactive CRP Visualization')
    parser.add_argument('--crp_data_dir', type=str, required=True,
                       help='Directory containing CRP analysis results')
    parser.add_argument('--output', type=str, default='crp_interactive_visualization.html',
                       help='Output HTML file path')

    args = parser.parse_args()

    print("Generating interactive CRP visualization...")
    generate_interactive_html(args.crp_data_dir, args.output)

if __name__ == "__main__":
    main()
