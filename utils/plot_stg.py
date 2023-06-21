import glob
import rasterio

import numpy as np
import torch

from skimage import transform

import plotly.graph_objects as go
import matplotlib as mpl

def plot(stg, label_names, label_colors, display="st", hover="none", show_gt="none", surface_ratio=4, tmin=0, tmax=23, node_color=None):
        r"""Plot an interactive figure of the spatio-temporal graph via plotly with pan and zoom tools
        and the possibility to see the object attributes by hovering.
        
        `label_names` is a list of the class names, such as `["impervious surface", "agriculture", "forest & other vegetation", "wetlands", "soil", "water", "snow & ice"]`.
        `label_colors` is a list of RGB colors to know in which color to display each class, i.e. `np.array([(96, 96, 96), (204, 204, 0), (0, 204, 0), (0, 0, 153), (153, 76, 0), (0, 128, 255), (138, 178, 198)])`.
        `display` sets the display of the different elements of the graph ['spatial', 'temporal, 'spatiotemporal', 'none'].
        `hover` allows to enable/disable the hovering information of the attributes ['spatial', 'temporal', 'region', 'all', 'none'].
        `tmin` and `tmax` adjust the temporal windows of the plot to limit the number of objects to display ; too much objects cause lags.
        `node_color` corresponds to an array of labels for nodes, and allows to display it with specific label especially the predicted ones. If set to `None`, the ground truth labels are used.
        `show_gt` allows to display a class map at the pixel resolution and for each date ['none', 'all', 'impervious surface', 'agriculture', ...].
        """

        assert len(label_names) == len(label_colors)

        # Compute the nodes and edges masks to display
        node_mask = torch.logical_and(tmin <= stg['region'].pos[:,2],stg['region'].pos[:,2] <= tmax)
        edge_spatial_mask = torch.isin(stg['region', 'spatial', 'region'].edge_index,stg['region'].node_index[node_mask])
        edge_spatial_mask = torch.logical_and(edge_spatial_mask[0], edge_spatial_mask[1])
        edge_temporal_mask = torch.isin(stg['region', 'temporal', 'region'].edge_index,stg['region'].node_index[node_mask])
        edge_temporal_mask = torch.logical_and(edge_temporal_mask[0], edge_temporal_mask[1])

        # Prepare the position data for Plotly
        coord = stg['region'].pos[node_mask]
        pos = dict(zip(stg['region'].node_index[node_mask].numpy(),coord))

        edge_spatial_xyz = stg['region', 'spatial', 'region'].edge_index[:,edge_spatial_mask].T.numpy()
        edge_temporal_xyz = stg['region', 'temporal', 'region'].edge_index[:,edge_temporal_mask].T.numpy()

        x_nodes= [pos[key][0]//surface_ratio for key in pos.keys()] # x-coordinates of nodes
        y_nodes = [pos[key][1]//surface_ratio for key in pos.keys()] # y-coordinates
        z_nodes = [pos[key][2] for key in pos.keys()] # z-coordinates

        # Create lists that contain the starting and ending coordinates of each edge.
        x_spatial_edges=[]
        y_spatial_edges=[]
        z_spatial_edges=[]
        x_temporal_edges=[]
        y_temporal_edges=[]
        z_temporal_edges=[]
        for edge in edge_spatial_xyz:
            #format: [beginning,ending,None]
            x_coords = [pos[edge[0]][0]//surface_ratio,pos[edge[1]][0]//surface_ratio,None]
            x_spatial_edges += x_coords
            y_coords = [pos[edge[0]][1]//surface_ratio,pos[edge[1]][1]//surface_ratio,None]
            y_spatial_edges += y_coords
            z_coords = [pos[edge[0]][2],pos[edge[1]][2],None]
            z_spatial_edges += z_coords
        for edge in edge_temporal_xyz:
            #format: [beginning,ending,None]
            x_coords = [pos[edge[0]][0]//surface_ratio,pos[edge[1]][0]//surface_ratio,None]
            x_temporal_edges += x_coords
            y_coords = [pos[edge[0]][1]//surface_ratio,pos[edge[1]][1]//surface_ratio,None]
            y_temporal_edges += y_coords
            z_coords = [pos[edge[0]][2],pos[edge[1]][2],None]
            z_temporal_edges += z_coords

        # Create a trace for the edges with hovering info
        trace_spatial_edges = go.Scatter3d(
            name="Spatial edges",
            x=x_spatial_edges,
            y=y_spatial_edges,
            z=z_spatial_edges,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo="all" if hover=="spatial" else "skip",
            hovertemplate="""
<b>avg sobel along boundary</b>: %{text[0]:r}<br>
<b>nb pixel along boundary</b>: %{text[1]}<extra></extra>""" if hover=="spatial" or hover=="all" else None,
            text = stg['region','spatial','region'].edge_attr[edge_spatial_mask].repeat_interleave(3,0))

        trace_temporal_edges = go.Scatter3d(
            name="Temporal edges",
            x=x_temporal_edges,
            y=y_temporal_edges,
            z=z_temporal_edges,
            mode='lines',
            line=dict(color='silver', width=1),
            hoverinfo="all" if hover=="temporal" else "skip",
            hovertemplate="""
<b>IoU between the connected regions</b>: %{text[0]:r}<extra></extra>""" if hover=="temporal" or hover=="all" else None,
            text = stg['region','temporal','region'].edge_attr[edge_temporal_mask].repeat_interleave(3,0))

        # Create a trace for the nodes that will be display with label colors and hovering info
        cmap = [ (i/6, mpl.colors.rgb2hex(label_colors[i,:]/255)) for i in range(label_colors.shape[0]) ]
        color = node_color
        if node_color is None:
            y = stg['region'].y[node_mask]
            if y.dim()>1:
                color = y.argmax(-1) # Take the majority class of the region as label
            else:
                color = y
        trace_nodes = go.Scatter3d(
            name="Region nodes",
            x=x_nodes,
            y=y_nodes,
            z=z_nodes,
            mode='markers',
            marker=dict(symbol='circle',
                    size=5,
                    color=color,
                    cmin=0,
                    cmax=6,
                    colorscale=cmap,
                    colorbar=dict(len=.5,
                                showticklabels=True,
                                tickmode="array",
                                ticktext=label_names,
                                tickvals=list(range(7)))
                    ),
            hoverinfo="all" if hover=="region" else "skip",
            hovertemplate="""
<b>blue mean</b>: %{text[0]:r}<br>
<b>blue std</b>: %{text[4]:r}<br>
<b>blue min</b>: %{text[12]}<br>
<b>blue max</b>: %{text[8]}<br>
<b>green mean</b>: %{text[1]:r}<br>
<b>green std</b>: %{text[5]:r}<br>
<b>green min</b>: %{text[13]}<br>
<b>green max</b>: %{text[9]}<br>
<b>red mean</b>: %{text[2]:r}<br>
<b>red std</b>: %{text[6]:r}<br>
<b>red min</b>: %{text[14]}<br>
<b>red max</b>: %{text[10]}<br>
<b>nir mean</b>: %{text[3]:r}<br>
<b>nir std</b>: %{text[7]:r}<br>
<b>nir min</b>: %{text[15]}<br>
<b>nir max</b>: %{text[11]}<br>
<b>mass center-x</b>: %{text[16]:r}<br>
<b>mass center-y</b>: %{text[17]:r}<extra></extra>""" if hover=="region" or hover=="all" else None,
            text = stg['region'].x[node_mask])
        
        # Create a trace to show gt/superpixels at each nodes layer
        if show_gt=="all" or show_gt in label_names:
            gt_files = sorted(glob.glob(stg.label_dir+"*.tif"))

            trace_gts = []
            for i in range(tmin,min(tmax+1,len(gt_files))):
                with rasterio.open(gt_files[i], 'r') as tif:
                    gt = tif.read().transpose((1,2,0))
                w, h, _ = gt.shape
                if show_gt in label_names:
                    cmin = 0
                    cmax = 255
                    cmap = [(0.,'#000000'),(1.,'#ffffff')]
                    gt_low = transform.downscale_local_mean(gt[:,:,label_names.index(show_gt)],surface_ratio)
                else:
                    cmin = 0
                    cmax = 6
                    cmap = [ (i/6, mpl.colors.rgb2hex(label_colors[i,:]/255)) for i in range(label_colors.shape[0]) ]
                    gt_low = transform.downscale_local_mean(gt.argmax(-1),surface_ratio)
                
                trace_gt = go.Surface(
                    x=np.linspace(0,w//surface_ratio, w//surface_ratio),
                    y=np.linspace(0,h//surface_ratio, h//surface_ratio),
                    z=np.full((w//surface_ratio,h//surface_ratio),i)-0.001,
                    surfacecolor=gt_low.T,
                    cmin=cmin,
                    cmax=cmax,
                    colorscale=cmap,
                    showscale=False,
                    hoverinfo="none")
                trace_gts.append(trace_gt)

        # Include the traces we want to plot and create a figure
        if str.lower(display)=="spatial" or str.lower(display)=="s":
            traces=[trace_nodes, trace_spatial_edges]
        elif str.lower(display)=="temporal" or str.lower(display)=="t":
            traces=[trace_nodes, trace_temporal_edges]
        elif str.lower(display)=="spatiotemporal" or str.lower(display)=="st":
            traces=[trace_nodes, trace_spatial_edges, trace_temporal_edges]
        elif str.lower(display)=="region" or str.lower(display)=="r":
            traces=[trace_nodes]
        else:
            traces=[]
        if show_gt!="none":
            traces = traces + trace_gts
        fig = go.Figure(data=traces)
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            scene = dict(
                zaxis = dict(range=[tmin-1,tmax+1],),
            )
        )
        fig.show()