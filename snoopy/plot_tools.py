import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from numpy import matlib

def plot_vector_field(pl, points, field, title='field', opacity=1.0, mag=1.0, cmap='jet', clim=[],
                      sym_xy=0, sym_yz=0, sym_xz=0):
    '''Plot a vector field in pyvista.
    
    :param pl:
        A pyvista plotter object.

    :param:
        The points in a (Mx3) numpy array.

    :param:
        The fields in a (Mx3) numpy array.        

    :param tite:
        The title for the colorbar,

    :param opacity:
        The opacity.

    :param mag:
        The scale parameter.

    :param cmap:
        The colormap.

    :param sym_xy:
        The symmetry condition on the xy plane.
        0 = none
        1 = normal field vanishes
        2 = tangential field vanishes

    :param sym_yz:
        The symmetry condition on the yz plane.
        0 = none
        1 = normal field vanishes
        2 = tangential field vanishes

    :param sym_xz:
        The symmetry condition on the xz plane.
        0 = none
        1 = normal field vanishes
        2 = tangential field vanishes

    :return:
        None.
    '''


    if sym_xy == 1:
        points_c = points.copy()
        field_c = field.copy()

        points_c[:, 2] *= -1.0
        field_c[:, 2] *= -1.0

        points = np.append(points, points_c, axis=0)
        field = np.append(field, field_c, axis=0)

    elif sym_xy == 2:
        points_c = points.copy()
        field_c = field.copy()

        points_c[:, 2] *= -1.0
        field_c[:, 0] *= -1.0
        field_c[:, 1] *= -1.0

        points = np.append(points, points_c, axis=0)
        field = np.append(field, field_c, axis=0)

    if sym_yz == 1:
        points_c = points.copy()
        field_c = field.copy()

        points_c[:, 0] *= -1.0
        field_c[:, 0] *= -1.0

        points = np.append(points, points_c, axis=0)
        field = np.append(field, field_c, axis=0)

    elif sym_yz == 2:
        points_c = points.copy()
        field_c = field.copy()

        points_c[:, 0] *= -1.0
        field_c[:, 1] *= -1.0
        field_c[:, 2] *= -1.0

        points = np.append(points, points_c, axis=0)
        field = np.append(field, field_c, axis=0)


    if sym_xz == 1:
        points_c = points.copy()
        field_c = field.copy()

        points_c[:, 1] *= -1.0
        field_c[:, 1] *= -1.0

        points = np.append(points, points_c, axis=0)
        field = np.append(field, field_c, axis=0)

    elif sym_xz == 2:
        points_c = points.copy()
        field_c = field.copy()

        points_c[:, 1] *= -1.0
        field_c[:, 0] *= -1.0
        field_c[:, 2] *= -1.0

        points = np.append(points, points_c, axis=0)
        field = np.append(field, field_c, axis=0)


    # the scalars for the arrows
    scalars_field = matlib.repmat(np.linalg.norm(field, axis=1), 15, 1).T.flatten()

    if len(clim) == 0:
        pl.add_arrows(points, field,
                        mag=mag,
                        cmap=cmap,
                        scalars=scalars_field,
                        scalar_bar_args={"title": title, "color": 'k'},
                        opacity=opacity)
    else:
        pl.add_arrows(points, field,
                        mag=mag,
                        cmap=cmap,
                        scalars=scalars_field,
                        scalar_bar_args={"title": title, "color": 'k'},
                        opacity=opacity,
                        clim=clim)


def plot_domain(pl, gmsh_mesh, domain_tag, color=[0.95, 0.95, 0.95],
                metallic=0.1,
                roughness=0.1,
                opacity=1.0,
                plot_volume=True,
                plot_feature_edges=True,
                show_edges=False,
                reflect_yz=False,
                reflect_xz=False,
                reflect_xy=False):
    '''Plot a domain in pyvista.

    :param pl:
        The pyvista plotter object.

    :param gmsh_mesh:
        The gmsh_mesh object.

    :param domain_tag:
        The domain tag to plot.

    :param color:
        The plot color.

    :param metallic:
        The metallic option for pyvista.

    :param roughness:
        The roughness option for pyvista.

    :param opacity:
        The plot opacity.
        
    :param plot_volume:
        Set this flag to true if You like to plot the volume.

    :param plot_feature edges:
        Set this flag to true if You like to plot the feature edges.

    :param show_edges:
        Set this to true in order to show the edged of the mesh.
    
    :param reflect_yz:
        Set this to true in order to reflect the domain in the yz plane.

    :param reflect_xz:
        Set this to true in order to reflect the domain in the xz plane.

    :param reflect_xy:
        Set this to true in order to reflect the domain in the xz plane.

    :return:
        None.
    '''

    node_tags, coord, parametricCoord = gmsh_mesh.getNodes() 
    elementTypes, elementTags, cells = gmsh_mesh.getElements(3, domain_tag) 

    # The nodes are not sorted correctly. I don't know why...
    # But we need to get them like this:
    num_nodes = len(node_tags)

    node_tags = np.unique(node_tags)

    # we now make an array of unique mesh nodes.
    nodes = np.zeros((num_nodes, 3))

    # the number of elements
    num_el = len(elementTags[0])

    nodes = coord
    nodes.shape = (np.int64(len(nodes)/3), 3)

    cells = cells[0] - 1
    cells.shape = (num_el, 4)


    # make the mesh object
    cell_info = np.append(np.ones((len(cells), 1), dtype=np.int64)*4, np.int64(cells), axis=1)
    mesh_list = [pv.UnstructuredGrid(cell_info, [pv.CellType.TETRA]*cells.shape[0], nodes)]

    # apply symmetry
    if reflect_yz:
        mesh_list.append(mesh_list[-1].reflect((1, 0, 0), point=(0, 0, 0)))

    if reflect_xz:
        for i in range(len(mesh_list)):
            mesh_list.append(mesh_list[i].reflect((0, 1, 0), point=(0, 0, 0)))

    if reflect_xy:
        for i in range(len(mesh_list)):
            mesh_list.append(mesh_list[i].reflect((0, 0, 1), point=(0, 0, 0)))

    for msh in mesh_list:

        dom_surf = msh.extract_surface()
        dom_surf.compute_normals(inplace=True, split_vertices=True)

        if plot_volume:
            pl.add_mesh(dom_surf, show_edges=show_edges, color=color, pbr=True, metallic=metallic, roughness=roughness, opacity=opacity)
        
        if plot_feature_edges:
            dom_edges = dom_surf.extract_feature_edges(45.)
            pl.add_mesh(dom_edges, color='black', line_width=2, opacity=1.0)
    

    return  