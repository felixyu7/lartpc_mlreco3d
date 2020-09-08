import plotly.graph_objs as go
import numpy as np


# coords format: [x1, y1, z1, x2, y2, z2]
def scatter_cuboid(coords, cubesize=1, colors='orange', hovertext=None, colorscale=None, opacity=0.2):
        
    trace = []
    trace.append(go.Mesh3d(
        # 8 vertices of a cube
#             x=[coords[0], coords[0], coords[0], coords[0], coords[3], coords[3], coords[3], coords[3]],
#             y=[coords[1], coords[1], coords[4], coords[4], coords[1], coords[1], coords[4], coords[4]],
#             z=[coords[2], coords[5], coords[2], coords[5], coords[2], coords[5], coords[2], coords[5]],
        x=[coords[0], coords[0], coords[0], coords[0]],
        y=[coords[1], coords[4], coords[1], coords[4]],
        z=[coords[2], coords[2], coords[5], coords[5]],
        opacity=opacity,
        color = colors,
        # i, j and k give the vertices of triangles
        delaunayaxis = 'x',
        showscale=True
    ))
    trace.append(go.Mesh3d(
            x=[coords[3], coords[3], coords[3], coords[3]],
            y=[coords[1], coords[4], coords[1], coords[4]],
            z=[coords[2], coords[2], coords[5], coords[5]],
            opacity=opacity,
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'x',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[1], coords[1], coords[1], coords[1]],
            z=[coords[2], coords[2], coords[5], coords[5]],
            opacity=opacity,
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'y',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[4], coords[4], coords[4], coords[4]],
            z=[coords[2], coords[2], coords[5], coords[5]],
            opacity=opacity,
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'y',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[1], coords[1], coords[4], coords[4]],
            z=[coords[2], coords[2], coords[2], coords[2]],
            opacity=opacity,
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'z',
            showscale=True
        ))
    trace.append(go.Mesh3d(
            x=[coords[0], coords[3], coords[0], coords[3]],
            y=[coords[1], coords[1], coords[4], coords[4]],
            z=[coords[5], coords[5], coords[5], coords[5]],
            opacity=opacity,
            color = colors,
            # i, j and k give the vertices of triangles
            delaunayaxis = 'z',
            showscale=True
        ))
    return trace