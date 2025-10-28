#!/usr/bin/env python3
"""
Generate Enhanced Interactive CRP Visualization
================================================

Features:
1. Multi-hop path highlighting (follows through all layers)
2. Encoder path included (bottleneck → encoder_1)
3. Skip connections visualization
4. Feature map display on hover/click
5. Adjustable top-K for each hop

Author: Claude Code
Date: October 28, 2025
"""

import json
import numpy as np
from pathlib import Path
import argparse
import base64

def generate_enhanced_html(crp_data_dir, output_html_path):
    """Generate enhanced interactive HTML visualization"""

    crp_data_dir = Path(crp_data_dir)
    metadata_path = crp_data_dir / 'metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Prepare images data
    images_data = {}
    for img_name, img_info in metadata['images'].items():
        print(f"Processing {img_name}...")

        transitions = {}
        for trans_key, trans_info in img_info['transitions'].items():
            matrix_path = crp_data_dir / trans_info['file']
            rel_matrix = np.load(matrix_path)
            transitions[trans_key] = rel_matrix.tolist()

        images_data[img_name] = {
            'tile_position': img_info['tile_position'],
            'transitions': transitions,
            'feature_maps': img_info.get('feature_maps', {})
        }

    # Generate HTML with embedded JavaScript
    html_content = generate_html_template(metadata, images_data, crp_data_dir)

    # Write HTML file
    with open(output_html_path, 'w') as f:
        f.write(html_content)

    print(f"\n✓ Enhanced visualization generated: {output_html_path}")

def generate_html_template(metadata, images_data, data_dir):
    """Generate complete HTML template"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced U-Net CRP Visualization - Multi-Hop Paths</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
            overflow-x: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            position: sticky;
            top: 0;
            z-index: 100;
        }}

        .header h1 {{
            margin: 0;
            font-size: 26px;
        }}

        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.95;
            font-size: 14px;
        }}

        .main-container {{
            display: grid;
            grid-template-columns: 300px 1fr 350px;
            gap: 20px;
            padding: 20px;
            min-height: calc(100vh - 80px);
        }}

        .sidebar {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            height: fit-content;
            position: sticky;
            top: 100px;
        }}

        .sidebar h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 18px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}

        .control-group {{
            margin-bottom: 20px;
        }}

        .control-group label {{
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
            font-size: 14px;
        }}

        select, input[type="range"] {{
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            background: white;
        }}

        input[type="range"] {{
            padding: 0;
            height: 8px;
        }}

        .slider-value {{
            text-align: center;
            font-weight: bold;
            color: #667eea;
            font-size: 20px;
            margin-top: 5px;
        }}

        button {{
            width: 100%;
            padding: 12px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: background 0.2s;
        }}

        button:hover {{
            background: #5568d3;
        }}

        .visualization {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            position: relative;
        }}

        #graph {{
            width: 100%;
            height: 900px;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            background: #fafafa;
        }}

        .info-panel {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            padding: 20px;
            height: fit-content;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
            position: sticky;
            top: 100px;
        }}

        .info-panel h3 {{
            margin-top: 0;
            color: #667eea;
            font-size: 18px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}

        #selectedNodeInfo {{
            color: #666;
            line-height: 1.8;
            font-size: 14px;
        }}

        .feature-map-preview {{
            margin-top: 15px;
            border-top: 1px solid #eee;
            padding-top: 15px;
        }}

        .feature-map-preview h4 {{
            color: #333;
            margin-bottom: 10px;
            font-size: 14px;
        }}

        .feature-map-preview img {{
            width: 100%;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
            stroke-width: 2.5px;
        }}

        .node-text {{
            font-size: 10px;
            font-weight: 600;
            pointer-events: none;
            user-select: none;
            fill: white;
        }}

        .link {{
            fill: none;
            stroke: #999;
            stroke-opacity: 0.2;
            stroke-width: 1px;
            transition: all 0.3s;
        }}

        .link-skip {{
            stroke: #FF6B6B;
            stroke-opacity: 0.15;
            stroke-dasharray: 5,5;
        }}

        .link-highlighted {{
            stroke: #FF4757 !important;
            stroke-width: 3px !important;
            stroke-opacity: 0.9 !important;
            stroke-dasharray: none !important;
        }}

        .link-highlighted-skip {{
            stroke: #FFA502 !important;
            stroke-width: 2.5px !important;
            stroke-opacity: 0.8 !important;
        }}

        .node-highlighted {{
            filter: brightness(1.4) drop-shadow(0 0 8px rgba(102, 126, 234, 0.6));
        }}

        .node-source {{
            filter: brightness(1.3) drop-shadow(0 0 6px rgba(255, 71, 87, 0.5));
        }}

        .layer-label {{
            font-size: 13px;
            font-weight: bold;
            fill: #333;
            text-anchor: middle;
        }}

        .path-info {{
            background: #f0f7ff;
            padding: 12px;
            border-radius: 6px;
            margin-top: 12px;
            border-left: 3px solid #667eea;
        }}

        .path-step {{
            margin: 8px 0;
            padding-left: 12px;
            border-left: 2px solid #ddd;
        }}

        .path-step strong {{
            color: #667eea;
        }}

        .relevance-badge {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 5px;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 15px 0;
        }}

        .stat-box {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 20px;
            font-weight: bold;
            color: #667eea;
        }}

        .stat-label {{
            font-size: 11px;
            color: #666;
            margin-top: 4px;
        }}

        .legend {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            margin-top: 15px;
            border: 1px solid #e0e0e0;
        }}

        .legend-title {{
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin: 8px 0;
            font-size: 12px;
        }}

        .legend-color {{
            width: 24px;
            height: 16px;
            margin-right: 8px;
            border-radius: 3px;
            border: 1px solid rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Enhanced U-Net CRP Visualization</h1>
        <p>Multi-Hop Path Tracing | Encoder Connections | Skip Connections | Feature Maps</p>
    </div>

    <div class="main-container">
        <!-- Left Sidebar: Controls -->
        <div class="sidebar">
            <h3>⚙️ Controls</h3>

            <div class="control-group">
                <label for="imageSelect">📷 Test Image:</label>
                <select id="imageSelect"></select>
            </div>

            <div class="control-group">
                <label for="startLayer">🎯 Start Layer:</label>
                <select id="startLayer">
                    <option value="decoder_1_conv2">Decoder 1</option>
                    <option value="decoder_2_conv2">Decoder 2</option>
                    <option value="decoder_3_conv2">Decoder 3</option>
                    <option value="decoder_4_conv2">Decoder 4</option>
                    <option value="bottleneck_conv2">Bottleneck</option>
                </select>
            </div>

            <div class="control-group">
                <label for="topK">🔝 Top-K Paths:</label>
                <input type="range" id="topK" min="1" max="10" value="2" step="1">
                <div class="slider-value" id="topKValue">2</div>
            </div>

            <div class="control-group">
                <label for="includeSkip">
                    <input type="checkbox" id="includeSkip" checked>
                    Show Skip Connections
                </label>
            </div>

            <div class="control-group">
                <button id="resetBtn">🔄 Reset View</button>
            </div>

            <div class="legend">
                <div class="legend-title">Layer Colors</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50;"></div>
                    <span>Decoder 1 (32ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #2196F3;"></div>
                    <span>Decoder 2 (64ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #9C27B0;"></div>
                    <span>Decoder 3 (128ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FF9800;"></div>
                    <span>Decoder 4 (256ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F44336;"></div>
                    <span>Bottleneck (512ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #00BCD4;"></div>
                    <span>Encoder 4 (256ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #3F51B5;"></div>
                    <span>Encoder 3 (128ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #673AB7;"></div>
                    <span>Encoder 2 (64ch)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #E91E63;"></div>
                    <span>Encoder 1 (32ch)</span>
                </div>
            </div>
        </div>

        <!-- Center: Visualization -->
        <div class="visualization">
            <svg id="graph"></svg>
        </div>

        <!-- Right Sidebar: Info Panel -->
        <div class="info-panel">
            <h3>📊 Node Information</h3>
            <div id="selectedNodeInfo">
                Click or hover on a node to see multi-hop paths and feature maps
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const metadata = {json.dumps(metadata, indent=2)};
        const imagesData = {json.dumps(images_data, indent=2)};
        const dataDir = "{str(data_dir)}";

        // Configuration
        const layerColors = {{
            'decoder_1_conv2': '#4CAF50',
            'decoder_2_conv2': '#2196F3',
            'decoder_3_conv2': '#9C27B0',
            'decoder_4_conv2': '#FF9800',
            'bottleneck_conv2': '#F44336',
            'encoder_4_conv2': '#00BCD4',
            'encoder_3_conv2': '#3F51B5',
            'encoder_2_conv2': '#673AB7',
            'encoder_1_conv2': '#E91E63'
        }};

        const layerChannels = metadata.layer_channels;
        const connections = metadata.connections;

        // UI elements
        const imageSelect = document.getElementById('imageSelect');
        const startLayerSelect = document.getElementById('startLayer');
        const topKSlider = document.getElementById('topK');
        const topKValue = document.getElementById('topKValue');
        const includeSkipCheckbox = document.getElementById('includeSkip');
        const resetBtn = document.getElementById('resetBtn');
        const selectedNodeInfo = document.getElementById('selectedNodeInfo');

        // Populate image selector
        Object.keys(imagesData).forEach(imgName => {{
            const option = document.createElement('option');
            option.value = imgName;
            option.textContent = imgName;
            imageSelect.appendChild(option);
        }});

        // State
        let currentImage = Object.keys(imagesData)[0];
        let currentStartLayer = 'decoder_1_conv2';
        let currentTopK = 2;
        let currentHighlightedNode = null;
        let includeSkipConnections = true;

        // Event listeners
        topKSlider.addEventListener('input', (e) => {{
            currentTopK = parseInt(e.target.value);
            topKValue.textContent = currentTopK;
            if (currentHighlightedNode) {{
                highlightMultiHopPaths(currentHighlightedNode);
            }}
        }});

        imageSelect.addEventListener('change', (e) => {{
            currentImage = e.target.value;
            renderGraph();
        }});

        startLayerSelect.addEventListener('change', (e) => {{
            currentStartLayer = e.target.value;
            renderGraph();
        }});

        includeSkipCheckbox.addEventListener('change', (e) => {{
            includeSkipConnections = e.target.checked;
            renderGraph();
        }});

        resetBtn.addEventListener('click', () => {{
            clearHighlights();
            currentHighlightedNode = null;
            selectedNodeInfo.innerHTML = 'Click or hover on a node to see multi-hop paths and feature maps';
        }});

        // D3 setup
        const svg = d3.select('#graph');
        const width = document.getElementById('graph').clientWidth;
        const height = 900;

        svg.attr('width', width).attr('height', height);

        const g = svg.append('g');

        // Zoom
        const zoom = d3.zoom()
            .scaleExtent([0.3, 3])
            .on('zoom', (event) => {{
                g.attr('transform', event.transform);
            }});

        svg.call(zoom);

        // Build connection graph
        function buildConnectionGraph() {{
            const graph = {{}};
            connections.forEach(([target, source]) => {{
                if (!graph[target]) graph[target] = [];
                graph[target].push(source);
            }});
            return graph;
        }}

        const connectionGraph = buildConnectionGraph();

        // Multi-hop path finder
        function findMultiHopPaths(startLayer, startChannel, maxHops = 10) {{
            const data = imagesData[currentImage];
            const paths = [];

            function explore(layer, channel, path, hop) {{
                if (hop >= maxHops) return;

                // Get possible next layers
                const nextLayers = connectionGraph[layer] || [];

                for (const nextLayer of nextLayers) {{
                    // Skip if not including skip connections
                    if (!includeSkipConnections &&
                        ((layer.includes('decoder') && nextLayer.includes('encoder')))) {{
                        continue;
                    }}

                    const transKey = `${{layer}}_from_${{nextLayer}}`;
                    if (!data.transitions[transKey]) continue;

                    const relMatrix = data.transitions[transKey];
                    if (!relMatrix[channel]) continue;

                    const sourceRelevances = relMatrix[channel];

                    // Get top-K sources
                    const topSources = sourceRelevances
                        .map((rel, idx) => ({{ channel: idx, relevance: rel }}))
                        .sort((a, b) => b.relevance - a.relevance)
                        .slice(0, currentTopK);

                    topSources.forEach(src => {{
                        const newPath = [...path, {{
                            from: {{ layer: layer, channel: channel }},
                            to: {{ layer: nextLayer, channel: src.channel }},
                            relevance: src.relevance
                        }}];

                        paths.push(newPath);

                        // Continue exploring
                        explore(nextLayer, src.channel, newPath, hop + 1);
                    }});
                }}
            }}

            explore(startLayer, startChannel, [], 0);
            return paths;
        }}

        // Render graph
        function renderGraph() {{
            g.selectAll('*').remove();
            clearHighlights();
            currentHighlightedNode = null;

            const data = imagesData[currentImage];

            // Determine which layers to show
            const allLayers = metadata.layer_sequence_full;
            const startIdx = allLayers.indexOf(currentStartLayer);
            const relevantLayers = allLayers.slice(startIdx);

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

            // Create links from connections
            connections.forEach(([targetLayer, sourceLayer]) => {{
                if (!relevantLayers.includes(targetLayer) || !relevantLayers.includes(sourceLayer)) return;

                // Skip if not including skip connections
                if (!includeSkipConnections &&
                    ((targetLayer.includes('decoder') && sourceLayer.includes('encoder')))) {{
                    return;
                }}

                const transKey = `${{targetLayer}}_from_${{sourceLayer}}`;
                if (!data.transitions[transKey]) return;

                const relMatrix = data.transitions[transKey];
                const isSkip = targetLayer.includes('decoder') && sourceLayer.includes('encoder');

                relMatrix.forEach((sourceRelevances, targetCh) => {{
                    sourceRelevances.forEach((relevance, sourceCh) => {{
                        if (relevance > 0.001) {{
                            const sourceNode = nodes.find(n => n.layer === sourceLayer && n.channel === sourceCh);
                            const targetNode = nodes.find(n => n.layer === targetLayer && n.channel === targetCh);

                            if (sourceNode && targetNode) {{
                                links.push({{
                                    source: sourceNode,
                                    target: targetNode,
                                    relevance: relevance,
                                    isSkip: isSkip,
                                    id: `${{sourceNode.id}}_to_${{targetNode.id}}`
                                }});
                            }}
                        }}
                    }});
                }});
            }});

            // Draw layer labels
            g.selectAll('.layer-label')
                .data(relevantLayers)
                .enter()
                .append('text')
                .attr('class', 'layer-label')
                .attr('x', (d, i) => layerSpacing * (i + 1))
                .attr('y', 30)
                .text(d => d.replace('_conv2', '').replace('_', ' '));

            // Draw links
            const linkGroup = g.append('g');
            linkGroup.selectAll('.link')
                .data(links)
                .enter()
                .append('line')
                .attr('class', d => d.isSkip ? 'link link-skip' : 'link')
                .attr('id', d => d.id)
                .attr('x1', d => d.source.x)
                .attr('y1', d => d.source.y)
                .attr('x2', d => d.target.x)
                .attr('y2', d => d.target.y)
                .attr('stroke-width', d => Math.max(0.5, d.relevance * 3));

            // Draw nodes
            const nodeGroup = g.append('g');
            const node = nodeGroup.selectAll('.node')
                .data(nodes)
                .enter()
                .append('g')
                .attr('class', 'node')
                .attr('id', d => d.id)
                .attr('transform', d => `translate(${{d.x}},${{d.y}})`)
                .on('mouseenter', function(event, d) {{
                    highlightMultiHopPaths(d);
                    currentHighlightedNode = d;
                }})
                .on('click', function(event, d) {{
                    highlightMultiHopPaths(d);
                    currentHighlightedNode = d;
                }});

            node.append('circle')
                .attr('class', 'node-circle')
                .attr('r', 5)
                .attr('fill', d => d.color);

            node.append('text')
                .attr('class', 'node-text')
                .attr('dy', '0.35em')
                .attr('text-anchor', 'middle')
                .text(d => d.channel);
        }}

        // Highlight multi-hop paths
        function highlightMultiHopPaths(node) {{
            clearHighlights();

            // Find all multi-hop paths
            const paths = findMultiHopPaths(node.layer, node.channel);

            if (paths.length === 0) {{
                selectedNodeInfo.innerHTML = `
                    <strong>Selected:</strong> ${{node.layer.replace('_', ' ')}} Channel ${{node.channel}}<br><br>
                    <em>No outgoing connections from this layer.</em>
                `;
                return;
            }}

            // Highlight all nodes and links in paths
            const highlightedNodes = new Set([`${{node.layer}}_ch${{node.channel}}`]);
            const highlightedLinks = new Set();

            paths.forEach(path => {{
                path.forEach(step => {{
                    const targetId = `${{step.from.layer}}_ch${{step.from.channel}}`;
                    const sourceId = `${{step.to.layer}}_ch${{step.to.channel}}`;
                    const linkId = `${{sourceId}}_to_${{targetId}}`;

                    highlightedNodes.add(targetId);
                    highlightedNodes.add(sourceId);
                    highlightedLinks.add(linkId);
                }});
            }});

            // Apply highlights
            highlightedNodes.forEach(nodeId => {{
                d3.select(`#${{nodeId}}`).classed('node-highlighted', true);
            }});

            highlightedLinks.forEach(linkId => {{
                const link = d3.select(`#${{linkId}}`);
                if (link.classed('link-skip')) {{
                    link.classed('link-highlighted-skip', true);
                }} else {{
                    link.classed('link-highlighted', true);
                }}
            }});

            // Update info panel with multi-hop paths
            updateInfoPanel(node, paths);
        }}

        // Update info panel
        function updateInfoPanel(node, paths) {{
            let html = `
                <strong>Selected Node:</strong><br>
                <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0;">
                    ${{node.layer.replace('_', ' ').toUpperCase()}}<br>
                    <span style="font-size: 24px; font-weight: bold; color: #667eea;">Channel ${{node.channel}}</span>
                </div>

                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">${{paths.length}}</div>
                        <div class="stat-label">Total Paths</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">${{currentTopK}}</div>
                        <div class="stat-label">Top-K Per Hop</div>
                    </div>
                </div>

                <strong>Multi-Hop Paths:</strong>
                <div class="path-info">
            `;

            // Group paths by depth
            const pathsByDepth = {{}};
            paths.forEach(path => {{
                const depth = path.length;
                if (!pathsByDepth[depth]) pathsByDepth[depth] = [];
                pathsByDepth[depth].push(path);
            }});

            Object.keys(pathsByDepth).sort((a, b) => a - b).forEach(depth => {{
                html += `<div style="margin-top: 15px;">`;
                html += `<strong style="color: #667eea;">Depth ${{depth}} (${{pathsByDepth[depth].length}} paths):</strong>`;

                // Show first few paths of each depth
                pathsByDepth[depth].slice(0, 3).forEach((path, idx) => {{
                    html += `<div class="path-step">`;
                    const lastStep = path[path.length - 1];
                    html += `→ ${{lastStep.to.layer.replace('_', ' ')}} Ch${{lastStep.to.channel}}`;
                    html += `<span class="relevance-badge">${{lastStep.relevance.toFixed(4)}}</span>`;
                    html += `</div>`;
                }});

                if (pathsByDepth[depth].length > 3) {{
                    html += `<div style="color: #999; font-size: 12px; margin-top: 5px;">... and ${{pathsByDepth[depth].length - 3}} more</div>`;
                }}
                html += `</div>`;
            }});

            html += `</div>`;

            // Feature map preview
            const imageData = imagesData[currentImage];
            if (imageData.feature_maps && imageData.feature_maps[node.layer]) {{
                const featureMapPath = imageData.feature_maps[node.layer].channels[node.channel];
                if (featureMapPath) {{
                    html += `
                        <div class="feature-map-preview">
                            <h4>Feature Map Preview:</h4>
                            <img src="${{dataDir}}/${{currentImage}}/${{featureMapPath}}" alt="Feature Map">
                        </div>
                    `;
                }}
            }}

            selectedNodeInfo.innerHTML = html;
        }}

        function clearHighlights() {{
            d3.selectAll('.link').classed('link-highlighted', false).classed('link-highlighted-skip', false);
            d3.selectAll('.node').classed('node-highlighted', false).classed('node-source', false);
        }}

        // Initial render
        renderGraph();
    </script>
</body>
</html>
"""

def main():
    parser = argparse.ArgumentParser(description='Generate Enhanced Interactive Visualization')
    parser.add_argument('--crp_data_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default='crp_enhanced_visualization.html')

    args = parser.parse_args()

    print("Generating enhanced interactive visualization...")
    generate_enhanced_html(args.crp_data_dir, args.output)

if __name__ == "__main__":
    main()
