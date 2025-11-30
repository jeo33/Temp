import numpy as np
import os
from read_data import *
from sklearn.decomposition import PCA
import json


def euclidean_to_centroid(query, centroids):
    return np.linalg.norm(centroids - query, axis=1)


print(os.getcwd())

dir = f'..\\data\\sift\\1000_20\\partitions'
groud_dir = f'..\\data\\sift\\sift_groundtruth.ivecs'
query_path = f'..\\data\\sift\\sift_query.fvecs'
base_path = f'..\\data\\sift\\sift_base.fvecs'
stats_dir = f'..\\data\\sift\\1000_20\\cluster_stats'

g = inspect_file_data(groud_dir)
q = inspect_file_data(query_path)
base = inspect_file_data(base_path)

# Load cluster data
centroids = []
all_ids = []
all_vecs = []
principal_components = []
eigenvalues_list = []

for it in range(1000):
    ids = inspect_file_data(os.path.join(dir, f'partition_{it}.ids'))
    vecs = inspect_file_data(os.path.join(dir, f'partition_{it}.fvecs'))
    all_ids.append(ids)
    all_vecs.append(vecs)

    centroids.append(np.load(os.path.join(stats_dir, f'centroid_{it}.npy')))
    principal_components.append(np.load(os.path.join(stats_dir, f'pca_{it}.npy')))
    eigenvalues_list.append(np.load(os.path.join(stats_dir, f'eigenvalues_{it}.npy')))

centroids = np.array(centroids)

# Build reverse index: vector_id -> cluster_id
id_to_cluster = {}
for cid in range(1000):
    for vid in all_ids[cid]:
        id_to_cluster[vid] = cid

# Output directory
output_dir = 'query_3d_viz'
os.makedirs(output_dir, exist_ok=True)


def generate_3d_html(qid):
    query = q[qid]
    gt_top10_ids = g[qid][:10]

    # Get distances and rankings
    euc_dists = euclidean_to_centroid(query, centroids)
    euc_ranking = np.argsort(euc_dists)
    top10_clusters = set(euc_ranking[:10])

    # Find GT clusters
    gt_clusters = set()
    gt_vectors = []
    gt_info = []
    for rank, gt_id in enumerate(gt_top10_ids):
        cid = id_to_cluster.get(gt_id)
        if cid is not None:
            gt_clusters.add(cid)
        gt_vectors.append(base[gt_id])
        gt_info.append({'rank': rank + 1, 'id': int(gt_id), 'cluster': int(cid) if cid else -1})
    gt_vectors = np.array(gt_vectors)

    # Clusters to visualize
    clusters_to_viz = list(top10_clusters | gt_clusters)

    # Collect all points for PCA
    all_points = [query]
    all_points.extend(gt_vectors)

    cluster_centroids_idx = {}
    for cid in clusters_to_viz:
        cluster_centroids_idx[cid] = len(all_points)
        all_points.append(centroids[cid])

    cluster_points_range = {}
    for cid in clusters_to_viz:
        start_idx = len(all_points)
        vecs = all_vecs[cid]
        # Sample for visualization
        if len(vecs) > 500:
            idx = np.random.choice(len(vecs), 500, replace=False)
            sampled = vecs[idx]
        else:
            sampled = vecs
        all_points.extend(sampled)
        cluster_points_range[cid] = (start_idx, len(all_points))

    all_points = np.array(all_points)

    # PCA to 3D
    pca = PCA(n_components=3)
    all_3d = pca.fit_transform(all_points)

    # Extract coordinates
    query_3d = all_3d[0].tolist()
    gt_3d = all_3d[1:1 + len(gt_vectors)].tolist()

    cluster_data = []
    for cid in clusters_to_viz:
        centroid_3d = all_3d[cluster_centroids_idx[cid]].tolist()
        start, end = cluster_points_range[cid]
        points_3d = all_3d[start:end].tolist()

        in_top10 = cid in top10_clusters
        has_gt = cid in gt_clusters
        euc_rank = int(np.sum(euc_dists < euc_dists[cid]) + 1)
        gt_count = sum(1 for gt_id in gt_top10_ids if id_to_cluster.get(gt_id) == cid)

        # Calculate gap
        vecs = all_vecs[cid]
        centroid = centroids[cid]
        diff = query - centroid
        diff_norm = diff / (np.linalg.norm(diff) + 1e-10)
        vecs_centered = vecs - centroid
        projections = np.dot(vecs_centered, diff_norm)
        max_proj = float(np.max(projections))
        gap = float(euc_dists[cid]) - max_proj

        # Compute ellipsoid parameters in 3D PCA space
        points_3d_arr = np.array(points_3d)
        if len(points_3d_arr) > 3:
            cov = np.cov(points_3d_arr.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = np.maximum(eigenvalues, 0.1)  # prevent zero
            radii = (2 * np.sqrt(eigenvalues)).tolist()  # 2 std
            eigenvectors = eigenvectors.tolist()
        else:
            radii = [10, 10, 10]
            eigenvectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        if in_top10 and has_gt:
            status = 'hit'
            color = '#00cc00'
        elif in_top10:
            status = 'fake'
            color = '#cc0000'
        else:
            status = 'missed'
            color = '#ff8800'

        cluster_data.append({
            'cid': int(cid),
            'centroid': centroid_3d,
            'points': points_3d,
            'in_top10': in_top10,
            'has_gt': has_gt,
            'status': status,
            'color': color,
            'euc_rank': euc_rank,
            'euc_dist': float(euc_dists[cid]),
            'gt_count': gt_count,
            'gap': gap,
            'radii': radii,
            'eigenvectors': eigenvectors
        })

    # Calculate summary stats
    total_gt_found = sum(1 for gt_id in gt_top10_ids if id_to_cluster.get(gt_id) in top10_clusters)
    n_fake = len(top10_clusters - gt_clusters)
    n_missed = len(gt_clusters - top10_clusters)

    # Generate HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query {qid} - 3D Cluster Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 20px;
        }}
        h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: rgba(255,255,255,0.1);
            padding: 10px 20px;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 1.5rem;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.8rem;
            opacity: 0.7;
        }}
        .stat.hit {{ border-left: 4px solid #00cc00; }}
        .stat.fake {{ border-left: 4px solid #cc0000; }}
        .stat.missed {{ border-left: 4px solid #ff8800; }}
        .main-content {{
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 20px;
        }}
        #plot3d {{
            width: 100%;
            height: 700px;
            background: #0a0a15;
            border-radius: 12px;
            border: 1px solid #333;
        }}
        .sidebar {{
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
            max-height: 700px;
            overflow-y: auto;
        }}
        .sidebar h3 {{
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }}
        .cluster-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;
            border-left: 4px solid;
            cursor: pointer;
            transition: transform 0.2s, background 0.2s;
        }}
        .cluster-card:hover {{
            transform: translateX(5px);
            background: rgba(255,255,255,0.1);
        }}
        .cluster-card.hit {{ border-color: #00cc00; }}
        .cluster-card.fake {{ border-color: #cc0000; }}
        .cluster-card.missed {{ border-color: #ff8800; }}
        .cluster-header {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }}
        .cluster-name {{
            font-weight: bold;
            font-size: 1.1rem;
        }}
        .cluster-status {{
            font-size: 0.75rem;
            padding: 2px 8px;
            border-radius: 4px;
            text-transform: uppercase;
        }}
        .cluster-status.hit {{ background: #00cc00; color: #000; }}
        .cluster-status.fake {{ background: #cc0000; color: #fff; }}
        .cluster-status.missed {{ background: #ff8800; color: #000; }}
        .cluster-details {{
            font-size: 0.85rem;
            opacity: 0.8;
        }}
        .cluster-details span {{
            display: inline-block;
            margin-right: 15px;
        }}
        .gt-section {{
            margin-top: 20px;
        }}
        .gt-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px;
            background: rgba(0, 255, 136, 0.1);
            border-radius: 4px;
            margin-bottom: 5px;
            font-size: 0.85rem;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.9rem;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 4px;
        }}
        .controls {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        .control-btn {{
            background: rgba(255,255,255,0.1);
            border: 1px solid #444;
            color: #fff;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .control-btn:hover {{
            background: rgba(255,255,255,0.2);
        }}
        .control-btn.active {{
            background: #00d4ff;
            color: #000;
        }}
        .nav-links {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
        }}
        .nav-links a {{
            color: #00d4ff;
            text-decoration: none;
            padding: 8px 16px;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
        }}
        .nav-links a:hover {{
            background: rgba(255,255,255,0.2);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Query {qid} - 3D Cluster Visualization</h1>
            <div class="stats">
                <div class="stat hit">
                    <div class="stat-value">{total_gt_found}/10</div>
                    <div class="stat-label">Recall@10</div>
                </div>
                <div class="stat hit">
                    <div class="stat-value">{len(gt_clusters) - n_missed}</div>
                    <div class="stat-label">HIT Clusters</div>
                </div>
                <div class="stat fake">
                    <div class="stat-value">{n_fake}</div>
                    <div class="stat-label">FAKE Clusters</div>
                </div>
                <div class="stat missed">
                    <div class="stat-value">{n_missed}</div>
                    <div class="stat-label">MISSED Clusters</div>
                </div>
            </div>
        </header>

        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #0066ff;"></div> Query</div>
            <div class="legend-item"><div class="legend-color" style="background: #00ff88;"></div> GT Neighbors</div>
            <div class="legend-item"><div class="legend-color" style="background: #00cc00;"></div> HIT Cluster</div>
            <div class="legend-item"><div class="legend-color" style="background: #cc0000;"></div> FAKE Cluster</div>
            <div class="legend-item"><div class="legend-color" style="background: #ff8800;"></div> MISSED Cluster</div>
        </div>

        <div class="controls">
            <button class="control-btn active" onclick="togglePoints(true)">Show Points</button>
            <button class="control-btn" onclick="togglePoints(false)">Hide Points</button>
            <button class="control-btn" onclick="toggleEllipsoids(true)">Show Boundaries</button>
            <button class="control-btn" onclick="toggleEllipsoids(false)">Hide Boundaries</button>
            <button class="control-btn" onclick="resetCamera()">Reset Camera</button>
        </div>

        <div class="main-content">
            <div id="plot3d"></div>
            <div class="sidebar">
                <h3>Clusters</h3>
                <div id="cluster-list"></div>

                <div class="gt-section">
                    <h3>Ground Truth Neighbors</h3>
                    <div id="gt-list"></div>
                </div>
            </div>
        </div>

        <div class="nav-links">
            <a href="query_{max(0, qid - 1)}_3d.html">← Previous Query</a>
            <a href="index.html">Index</a>
            <a href="query_{min(9, qid + 1)}_3d.html">Next Query →</a>
        </div>
    </div>

    <script>
        // Data
        const queryPoint = {json.dumps(query_3d)};
        const gtPoints = {json.dumps(gt_3d)};
        const gtInfo = {json.dumps(gt_info)};
        const clusters = {json.dumps(cluster_data)};

        let showPoints = true;
        let showEllipsoids = false;

        function createPlot() {{
            const traces = [];

            // Query point
            traces.push({{
                x: [queryPoint[0]],
                y: [queryPoint[1]],
                z: [queryPoint[2]],
                mode: 'markers',
                type: 'scatter3d',
                name: 'Query',
                marker: {{
                    size: 12,
                    color: '#0066ff',
                    symbol: 'diamond',
                    line: {{ color: '#fff', width: 2 }}
                }},
                hovertemplate: '<b>QUERY</b><extra></extra>'
            }});

            // GT points
            traces.push({{
                x: gtPoints.map(p => p[0]),
                y: gtPoints.map(p => p[1]),
                z: gtPoints.map(p => p[2]),
                mode: 'markers+text',
                type: 'scatter3d',
                name: 'GT Neighbors',
                marker: {{
                    size: 8,
                    color: '#00ff88',
                    symbol: 'square',
                    line: {{ color: '#000', width: 1 }}
                }},
                text: gtInfo.map(g => g.rank.toString()),
                textposition: 'top center',
                textfont: {{ size: 10, color: '#fff' }},
                hovertemplate: gtInfo.map(g => 
                    `<b>GT #${{g.rank}}</b><br>ID: ${{g.id}}<br>Cluster: ${{g.cluster}}<extra></extra>`
                )
            }});

            // Lines from query to GT
            gtPoints.forEach((gt, i) => {{
                traces.push({{
                    x: [queryPoint[0], gt[0]],
                    y: [queryPoint[1], gt[1]],
                    z: [queryPoint[2], gt[2]],
                    mode: 'lines',
                    type: 'scatter3d',
                    name: '',
                    showlegend: false,
                    line: {{ color: 'rgba(0, 255, 136, 0.3)', width: 1 }},
                    hoverinfo: 'skip'
                }});
            }});

            // Clusters
            clusters.forEach(cluster => {{
                // Cluster points
                if (showPoints) {{
                    traces.push({{
                        x: cluster.points.map(p => p[0]),
                        y: cluster.points.map(p => p[1]),
                        z: cluster.points.map(p => p[2]),
                        mode: 'markers',
                        type: 'scatter3d',
                        name: `C${{cluster.cid}} points`,
                        visible: showPoints ? true : 'legendonly',
                        marker: {{
                            size: 2,
                            color: cluster.color,
                            opacity: 0.4
                        }},
                        hovertemplate: `Cluster ${{cluster.cid}}<extra></extra>`
                    }});
                }}

                // Centroid
                traces.push({{
                    x: [cluster.centroid[0]],
                    y: [cluster.centroid[1]],
                    z: [cluster.centroid[2]],
                    mode: 'markers+text',
                    type: 'scatter3d',
                    name: `C${{cluster.cid}} centroid`,
                    marker: {{
                        size: 10,
                        color: cluster.color,
                        symbol: 'x',
                        line: {{ color: '#fff', width: 1 }}
                    }},
                    text: [`C${{cluster.cid}}`],
                    textposition: 'top center',
                    textfont: {{ size: 10, color: '#fff' }},
                    hovertemplate: `<b>Cluster ${{cluster.cid}}</b><br>` +
                        `Status: ${{cluster.status.toUpperCase()}}<br>` +
                        `Rank: ${{cluster.euc_rank}}<br>` +
                        `Distance: ${{cluster.euc_dist.toFixed(1)}}<br>` +
                        `GT Count: ${{cluster.gt_count}}<br>` +
                        `Gap: ${{cluster.gap.toFixed(1)}}<extra></extra>`
                }});

                // Line from query to centroid
                traces.push({{
                    x: [queryPoint[0], cluster.centroid[0]],
                    y: [queryPoint[1], cluster.centroid[1]],
                    z: [queryPoint[2], cluster.centroid[2]],
                    mode: 'lines',
                    type: 'scatter3d',
                    name: '',
                    showlegend: false,
                    line: {{
                        color: cluster.color,
                        width: 2,
                        dash: 'dash'
                    }},
                    hoverinfo: 'skip'
                }});

                // Ellipsoid surface (approximate with mesh)
                if (showEllipsoids) {{
                    const ellipsoidTrace = createEllipsoid(
                        cluster.centroid, 
                        cluster.radii, 
                        cluster.eigenvectors,
                        cluster.color
                    );
                    traces.push(ellipsoidTrace);
                }}
            }});

            const layout = {{
                scene: {{
                    xaxis: {{ title: 'PC1', gridcolor: '#333', zerolinecolor: '#555' }},
                    yaxis: {{ title: 'PC2', gridcolor: '#333', zerolinecolor: '#555' }},
                    zaxis: {{ title: 'PC3', gridcolor: '#333', zerolinecolor: '#555' }},
                    bgcolor: '#0a0a15',
                    camera: {{
                        eye: {{ x: 1.5, y: 1.5, z: 1.5 }}
                    }}
                }},
                paper_bgcolor: '#0a0a15',
                plot_bgcolor: '#0a0a15',
                font: {{ color: '#eee' }},
                showlegend: false,
                margin: {{ l: 0, r: 0, t: 0, b: 0 }}
            }};

            Plotly.newPlot('plot3d', traces, layout, {{ responsive: true }});
        }}

        function createEllipsoid(center, radii, eigenvectors, color) {{
            const n = 20;
            const u = [];
            const v = [];
            for (let i = 0; i <= n; i++) {{
                u.push(i * Math.PI / n);
            }}
            for (let j = 0; j <= n; j++) {{
                v.push(j * 2 * Math.PI / n);
            }}

            const x = [], y = [], z = [];
            for (let i = 0; i < u.length; i++) {{
                const xRow = [], yRow = [], zRow = [];
                for (let j = 0; j < v.length; j++) {{
                    const px = radii[0] * Math.sin(u[i]) * Math.cos(v[j]);
                    const py = radii[1] * Math.sin(u[i]) * Math.sin(v[j]);
                    const pz = radii[2] * Math.cos(u[i]);

                    // Rotate by eigenvectors
                    const rx = eigenvectors[0][0]*px + eigenvectors[0][1]*py + eigenvectors[0][2]*pz;
                    const ry = eigenvectors[1][0]*px + eigenvectors[1][1]*py + eigenvectors[1][2]*pz;
                    const rz = eigenvectors[2][0]*px + eigenvectors[2][1]*py + eigenvectors[2][2]*pz;

                    xRow.push(center[0] + rx);
                    yRow.push(center[1] + ry);
                    zRow.push(center[2] + rz);
                }}
                x.push(xRow);
                y.push(yRow);
                z.push(zRow);
            }}

            return {{
                type: 'surface',
                x: x,
                y: y,
                z: z,
                opacity: 0.3,
                colorscale: [[0, color], [1, color]],
                showscale: false,
                hoverinfo: 'skip'
            }};
        }}

        function togglePoints(show) {{
            showPoints = show;
            document.querySelectorAll('.control-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            createPlot();
        }}

        function toggleEllipsoids(show) {{
            showEllipsoids = show;
            createPlot();
        }}

        function resetCamera() {{
            Plotly.relayout('plot3d', {{
                'scene.camera': {{ eye: {{ x: 1.5, y: 1.5, z: 1.5 }} }}
            }});
        }}

        function focusCluster(cid) {{
            const cluster = clusters.find(c => c.cid === cid);
            if (cluster) {{
                Plotly.relayout('plot3d', {{
                    'scene.camera': {{
                        eye: {{
                            x: cluster.centroid[0] / 50 + 1,
                            y: cluster.centroid[1] / 50 + 1,
                            z: cluster.centroid[2] / 50 + 1
                        }},
                        center: {{
                            x: cluster.centroid[0] / 100,
                            y: cluster.centroid[1] / 100,
                            z: cluster.centroid[2] / 100
                        }}
                    }}
                }});
            }}
        }}

        // Populate sidebar
        function populateSidebar() {{
            const clusterList = document.getElementById('cluster-list');
            const sortedClusters = [...clusters].sort((a, b) => a.euc_rank - b.euc_rank);

            sortedClusters.forEach(cluster => {{
                const card = document.createElement('div');
                card.className = `cluster-card ${{cluster.status}}`;
                card.onclick = () => focusCluster(cluster.cid);
                card.innerHTML = `
                    <div class="cluster-header">
                        <span class="cluster-name">Cluster ${{cluster.cid}}</span>
                        <span class="cluster-status ${{cluster.status}}">${{cluster.status}}</span>
                    </div>
                    <div class="cluster-details">
                        <span>Rank: ${{cluster.euc_rank}}</span>
                        <span>Dist: ${{cluster.euc_dist.toFixed(1)}}</span>
                        <span>GT: ${{cluster.gt_count}}</span>
                        <span>Gap: ${{cluster.gap.toFixed(1)}}</span>
                    </div>
                `;
                clusterList.appendChild(card);
            }});

            const gtList = document.getElementById('gt-list');
            gtInfo.forEach((gt, i) => {{
                const item = document.createElement('div');
                item.className = 'gt-item';
                const inTop10 = clusters.some(c => c.cid === gt.cluster && c.in_top10);
                item.innerHTML = `
                    <span>#${{gt.rank}} (ID: ${{gt.id}})</span>
                    <span>C${{gt.cluster}} ${{inTop10 ? '✓' : '✗'}}</span>
                `;
                gtList.appendChild(item);
            }});
        }}

        // Initialize
        createPlot();
        populateSidebar();
    </script>
</body>
</html>
'''
    return html


# Generate HTML for each query
for qid in range(10):
    html = generate_3d_html(qid)
    filepath = os.path.join(output_dir, f'query_{qid}_3d.html')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Generated: {filepath}")

# Generate index page
index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query 3D Visualizations - Index</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 40px;
        }
        h1 {
            text-align: center;
            margin-bottom: 40px;
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background: rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s, background 0.3s;
            text-decoration: none;
            color: #eee;
        }
        .card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.15);
        }
        .card h2 {
            margin-bottom: 10px;
            color: #00d4ff;
        }
        .card p {
            opacity: 0.8;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <h1>Query 3D Visualizations</h1>
    <div class="grid">
'''

for qid in range(10):
    query = q[qid]
    gt_top10_ids = g[qid][:10]
    euc_dists = euclidean_to_centroid(query, centroids)
    top10_clusters = set(np.argsort(euc_dists)[:10])

    gt_clusters = set()
    for gt_id in gt_top10_ids:
        cid = id_to_cluster.get(gt_id)
        if cid is not None:
            gt_clusters.add(cid)

    total_gt_found = sum(1 for gt_id in gt_top10_ids if id_to_cluster.get(gt_id) in top10_clusters)
    n_fake = len(top10_clusters - gt_clusters)
    n_missed = len(gt_clusters - top10_clusters)

    index_html += f'''
        <a href="query_{qid}_3d.html" class="card">
            <h2>Query {qid}</h2>
            <p>Recall: {total_gt_found}/10 | Fake: {n_fake} | Missed: {n_missed}</p>
        </a>
    '''

index_html += '''
    </div>
</body>
</html>
'''

with open(os.path.join(output_dir, 'index.html'), 'w', encoding='utf-8') as f:
    f.write(index_html)

print(f"\nGenerated index: {os.path.join(output_dir, 'index.html')}")
print(f"\nAll files saved to: {output_dir}/")