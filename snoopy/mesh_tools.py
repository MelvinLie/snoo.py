import numpy as np
import gmsh
import pyvista as pv
import matplotlib.pyplot as plt
import networkx as nx

def get_mesh_info(mesh, element_order=1):
    '''Get some information about the mesh.
    
    :param mesh:
        A gmsh mesh object.

    :param element_order:
        The order of the finite element.

    :return:
        The number of degrees of freedom (assuming a certain
        fem approximation space), the number of faces,
        the number of edges and the element tags.
    '''
    
    # get element information
    elementTypes, elementTags, elementNodes = mesh.getElements(dim=3)
    elementTypes = elementTypes[0]
    elementTags = elementTags[0]
    elementNodes = elementNodes[0]

    # number of elements
    num_elements = elementTags.shape[0]

    # get edge information
    mesh.createEdges()
    edgeTags, edgeNodes = mesh.getAllEdges()

    # number of edges
    num_edges = len(edgeTags)

    # get face information
    mesh.createFaces()

    # get faces
    faceTags, faceNodes = mesh.getAllFaces(3)

    # number of faces
    num_faces = len(faceTags)

    # process information
    
    if element_order == 1:
        num_dofs = num_edges
    elif element_order == 2:
        num_dofs = 2*num_edges
    elif element_order == 3:
        num_dofs = 3*num_edges + 4*num_faces + 6*num_elements
    elif element_order == 4:
        num_dofs = 3*num_edges + 4*num_faces + 6*num_elements
    else:
        print('Warning! Element order {} not implemented!')
        num_dofs = -1

    return num_dofs, num_faces, num_edges, elementTags

def get_vector_basis_mesh_info(mesh, loc):
    '''Get all mesh information needed for the matrix assembly using
    vectorial finite elements, for instance Hcurl conforming ones.

    :param mesh:
        A gmsh mesh object.

    :param int_order:
        The integration local coordinates.

    :return:
        The jacobians at the integration points,
        the determinants at the integration points,
        the physical coordinates at the integration points,
        the orientation of the finite elements.
    '''

    # get some mesh info
    _, _, _, elementTags = get_mesh_info(mesh)
    num_el = elementTags.shape[0]

    # number of integration points
    num_points = np.int64(len(loc)/3)
    # jacobians data container
    jacobians = np.zeros((num_el, num_points*9))
    # determinants container
    determinants = np.zeros((num_el, num_points))
    # coordinates container
    coordinates = np.zeros((num_el, num_points*3))
    # orientations container
    orientations = np.zeros((1, num_el), dtype=np.int32)

    # loop over the elements
    for e in range(num_el):

        # evaluate the finite elements
        jacobians[e, :], determinants[e, :], coordinates[e, :] = mesh.getJacobian(elementTags[e], loc)

        # get the orientation of this element
        orientations[0, e] = mesh.getBasisFunctionsOrientationForElement(elementTags[e], "CurlHcurlLegendre2")

    return jacobians, determinants, coordinates, orientations

def get_vector_basis_orientations(mesh, element_tags, element_spec='HcurlLegendre1'):
    '''Get all element orientations needed for the matrix assembly using
    vectorial finite elements, for instance Hcurl conforming ones.

    :param mesh:
        A gmsh mesh object.

    :param element_tags:
        The gmsh element tag.

    :param element_spec:
        The finite element gmsh specifyer.

    :return:
        the orientation of the finite elements.
    '''

    # get some mesh info
    num_el = len(element_tags)

    # orientations container
    orientations = np.zeros((num_el, ), dtype=np.int32)

    # loop over the elements
    for i in range(num_el):

        # get the orientation of this element
        orientations[i] = mesh.getBasisFunctionsOrientationForElement(element_tags[i], element_spec)

    return orientations

def get_num_edge_dofs(order, element_type='Tet'):
    '''Get the number of degrees of freedom per element for
    a certain finite element order.

    :param element_order:
        The order of the finite element.

    :param element_type:
        The finite element type. Currently only Tet is supported!

    :return:
        The number of dofs.
    '''

    if element_type == 'Tet':
        if order == 1:
            return 6
        elif order == 2:
            return 12
        elif order == 3:
            return 30
        else:
            print('Tet edge elements of order {} unsupported!'.format(order))
            return -1

    elif element_type == 'Hex':
        if order == 1:
            return 12
        elif order == 2:
            return 54
        elif order == 3:
            return 144
        else:
            print('Hex edge elements of order {} unknown!'.format(order))
            return -1
        
    else:
        print('Edge elements of type {} unknown!'.format(element_type))


def get_global_ids_for_entities(type_keys, entity_keys, num_el, num_edges, num_faces, order):
    '''Function to compute the global identification numbers of degrees of freedom.

    :param type_keys:
        The type keys.

    :param entity_keys:
        The entity keys.

    :param num_edges:
        The number of edges in the mesh.

    :param num_faces:
        The number of faces in the mesh.

    :param order:
        The order of the edge elements.

    :return:
        The global ids array.

    '''


    if order == 1:
        # all type keys are 1. i.e. edge functions
        return entity_keys

    elif order == 2:
        # type keys are:
        # 1: edge function (1st kind)
        # 2: edge function (2nd kind)

        return np.int32(entity_keys + (type_keys == 2)*num_edges)
        
    elif order == 3:
        # type keys are:
        # 1: edge function (1st kind)
        # 2: edge function (2nd kind)

        return np.int32(entity_keys + (type_keys == 2)*num_edges \
                + (type_keys == 3)*2*num_edges \
                + (type_keys == 4)*(2*num_edges + num_faces) \
                + (type_keys == 5)*(2*num_edges  + 2*num_faces)\
                + (type_keys == 6)*(2*num_edges  + 3*num_faces)\
                + (type_keys == 7)*(2*num_edges  + 4*num_faces)\
                + (type_keys == 8)*(2*num_edges  + 4*num_faces + num_el)\
                + (type_keys == 9)*(2*num_edges  + 4*num_faces + 2*num_el)\
                + (type_keys == 10)*(2*num_edges  + 4*num_faces + 3*num_el)\
                + (type_keys == 11)*(2*num_edges  + 4*num_faces + 4*num_el)\
                + (type_keys == 12)*(2*num_edges  + 4*num_faces + 5*num_el))

    elif order == 4:
        # type keys are:
        # 0: vertex function

        # 1: edge function (1st kind)
        # 2: edge function (2nd kind)

        # 3: face function (1st kind)
        # 4: face function (2nd kind)
        # 5: face function (3rd kind)
        # 6: face function (4th kind)

        # 7: bubble function (1st kind)
        # 8: bubble function (2nd kind)
        # 9: bubble function (3rd kind)
        # 10: bubble function (4th kind)
        # 11: bubble function (5th kind)
        # 12: bubble function (6th kind)

        return np.int32(entity_keys + (type_keys == 2)*num_edges \
                + (type_keys == 3)*2*num_edges \
                + (type_keys == 4)*(2*num_edges + num_faces) \
                + (type_keys == 5)*(2*num_edges  + 2*num_faces)\
                + (type_keys == 6)*(2*num_edges  + 3*num_faces)\
                + (type_keys == 7)*(2*num_edges  + 4*num_faces)\
                + (type_keys == 8)*(2*num_edges  + 4*num_faces + num_el)\
                + (type_keys == 9)*(2*num_edges  + 4*num_faces + 2*num_el)\
                + (type_keys == 10)*(2*num_edges  + 4*num_faces + 3*num_el)\
                + (type_keys == 11)*(2*num_edges  + 4*num_faces + 4*num_el)\
                + (type_keys == 12)*(2*num_edges  + 4*num_faces + 5*num_el))
        
    elif order == 5:
        # type keys are:
        # 0: vertex function

        # 1: edge function (1st kind)
        # 2: edge function (2nd kind)
        # 3: edge function (3rd kind)

        # 4: face function (1st kind)
        # 5: face function (2nd kind)
        # 6: face function (3rd kind)
        # 7: face function (4th kind)
        # 8: face function (5th kind)
        # 9: face function (6th kind)
        # 10: face function (7th kind)
        # 11: face function (8th kind)
        # 12: face function (9th kind)
        # 13: face function (10th kind)
        # 14: face function (11th kind)
        # 15: face function (12th kind)

        # 16: bubble function (1st kind)
        # 17: bubble function (2nd kind)
        # 18: bubble function (3rd kind)
        # 19: bubble function (4th kind)
        # 20: bubble function (5th kind)
        # ...
        # 
        # 51: bubble function (36th kind)
            
        ret_keys = entity_keys.copy()

        offset = num_edges
        # edge basis functions
        for type in range(2, 4):
            ret_keys[type_keys == type] += offset
            offset += num_edges

        # face basis functions
        for type in range(4, 16):
            ret_keys[type_keys == type] += offset
            offset += num_faces

        # bubble functions
        for type in range(16, 52):
            ret_keys[type_keys == type] += offset
            offset += num_el
            
        return ret_keys
        
    else:
        print('Edge elements of order {} are not implemented yet!'.format(order))
        return -1

def get_global_ids(mesh, element_order):
    '''Get the global ids for the edge basis functions of a certain order.

    :param mesh:
        A gmsh mesh object.

    :param element_order:
        The order of the finite element.

    :return:
        The global ids.
    '''
        
    # get some mesh info
    _, num_faces, num_edges, elementTags = get_mesh_info(mesh)
    num_el = elementTags.shape[0]

    # the number of dofs per element
    num_dof_el = get_num_edge_dofs(element_order)

    # global ids container
    global_ids = np.zeros((num_el, num_dof_el), dtype=np.int32)

    # this is the function type string
    function_type = 'HcurlLegendre' + str(element_order-1)

    # loop over the elements
    for e in range(num_el):

        #get the global ids
        typeKeys, entityKeys, _ = mesh.getKeysForElement(elementTags[e], function_type, returnCoord=False)

        global_ids[e, :] = get_global_ids_for_entities(typeKeys, entityKeys, num_el, num_edges, num_faces, element_order)

    return global_ids

def get_edge_boundary_dofs(model, boundary_names, element_order=1):
    '''Get edge elements boundary DoFs.

    :param model:
        A gmsh model object.

    :param boundary_names:
        The name tags of the Dirichlet boundaries.

    :return:
        The indices of the edge_dofs and face_dofs
    '''
    # create all edges and faces
    model.mesh.createEdges()
    model.mesh.createFaces()

    # get all physical groups corresponding to surfaces
    boundary_groups = model.getPhysicalGroups(2)

    # FOR THE FUTURE
    # check if they are Dirichlet or Neumann type
    # so far only Dirichlet is implemented

    # number of boundaries
    num_boundaries = len(boundary_groups)

    num_dofs, num_faces,num_edges, elementTags = get_mesh_info(model.mesh)
    num_el = elementTags.shape[0]

    # containers for edge and face DoFs
    boundary_dofs = []
    # edge_dofs = []
    # face_dofs = []

    #function_space_string = 'HcurlLegendre{}'.format(element_order-1)
    function_space_string = 'HcurlLegendre{}'.format(element_order)

    # function space string
    if element_order > 3:
        print('Element order {} not supported!'.format(element_order))
        
    for bn in boundary_names:

        # get the dim tags
        dim_tags = model.getEntitiesForPhysicalName(bn)

        # loop over the boundaries
        for i, dt in enumerate(dim_tags):

            # boundary faces info
            b_face_types, b_face_tags, b_face_nodes = model.mesh.getElements(dt[0], dt[1])

            # loop over the faces
            for j in range(len(b_face_tags[0])):

                # get the keys for the DoFs of this face
                b_face_type_keys, b_face_entity_keys, b_face_coord = model.mesh.getKeysForElement(b_face_tags[0][j], function_space_string)

                glob_ids = get_global_ids_for_entities(b_face_type_keys, b_face_entity_keys, num_el, num_edges, num_faces, element_order)

                # make global indices based on type and face key
                for k in range(len(b_face_type_keys)):

                    # construct the global id
                    # global_id = np.int32(b_face_entity_keys[k])
                    global_id = np.int32(glob_ids[k])

                    boundary_dofs.append(global_id)
                    # if b_face_type_keys[k] < 4:
                    #     edge_dofs.append(global_id)
                    # else:
                    #    face_dofs.append(global_id)

    # make them unique
    boundary_dofs = np.unique(np.array(boundary_dofs))
    # face_dofs = np.unique(np.array(face_dofs))

    return boundary_dofs


def setup_coil_terminals(model, domain_name, pos_terminals):
    '''Setup the terminals for a coil.

    :param model:
        The gmsh model.

    :param domain_name:
        The name of the coil domain.

    :param pos_terminals:
        The center positions of the two terminals.

    :return:
        None.
    '''

    # get the two terminals
    vol = model.getEntitiesForPhysicalName(domain_name)[0]
    
    # get the boundary of the coil domain
    boundary = model.getBoundary([vol])


    # loop over the boundary surfaces
    for i, bs in boundary:
        
        normal = model.getNormal(abs(bs), [0., 0.])

        diff = np.linalg.norm(np.array([1., 0., 0.]) - np.array([abs(normal[0]), normal[1], normal[2]]))

        if diff < 1e-12:
            if model.isInside(2, abs(bs), pos_terminals[0]):
                model.addPhysicalGroup(2, [abs(bs)], name=domain_name + '_terminal_1')
            elif model.isInside(2, abs(bs), pos_terminals[1]):
                model.addPhysicalGroup(2, [abs(bs)], name=domain_name + '_terminal_2')
    
    return 

def get_cotree_dofs(gmsh_mesh):
    '''Get the degrees of freedom of the co-treee.

    :param gmsh_mesh:
        The gmsh mesh object.

    :return:
        The degrees of freedom of the cotree.
    '''

    # make all edges
    gmsh_mesh.createEdges()

    # get the edge connectivity
    edge_tags, edge_nodes  = gmsh_mesh.get_all_edges()
    edge_nodes.shape = (len(edge_tags), 2)
    
    nodes = np.zeros((len(edge_tags), 3))

    # setup a graph object
    G = nx.Graph()
    
    # this map is needed to get back to the node tags
    edge_id_map = {}
    # add all edges
    for i, e in enumerate(edge_nodes):
        edge_tuple = tuple(sorted((e[0], e[1])))
        G.add_edge(edge_tuple[0], edge_tuple[1], weight=1)
        edge_id_map[edge_tuple] = edge_tags[i]
    
    # Ensure the graph is connected
    if not nx.is_connected(G):
        raise ValueError("The input graph must be connected.")

    # get the spanning tree
    spanning_tree = nx.maximum_spanning_tree(G,algorithm='boruvka')
    spanning_tree_edges = list(spanning_tree.edges())

    # get the co-tree
    all_edges = set(G.edges())
    tree_edges = set(spanning_tree_edges)
    cotree_edges = list(all_edges - tree_edges)


    print('number of tree edges = {}'.format(len(spanning_tree_edges)))
    print('number of co-tree edges = {}'.format(len(cotree_edges)))


    tree_edge_dofs = []
    for edge in list(tree_edges):
        tree_edge_dofs.append(edge_id_map[tuple(sorted((edge[0], edge[1])))])

    co_tree_edge_dofs = []
    for edge in list(all_edges - tree_edges):
        co_tree_edge_dofs.append(edge_id_map[tuple(sorted((edge[0], edge[1])))])

    tree_edge_dofs = np.array(tree_edge_dofs, dtype=np.int64) - 1
    co_tree_edge_dofs = np.array(co_tree_edge_dofs, dtype=np.int64) - 1

    return tree_edge_dofs, co_tree_edge_dofs