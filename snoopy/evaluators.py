import numpy as np
import gmsh

from .mesh_tools import get_vector_basis_orientations
from .finite_element import FiniteElement

def compute_grad_phi(c, x, d_phi, inv_J):
    '''Compute the magnetic field in a finite element
    with connectivity c. The vector x stores the nodal 
    values of the solution.
    The array grad_phi stores the gradients of the FEM
    basis functions.

    :param c:
        The node connectivity of the finite element.

    :param x:
        The solution vector.

    :param d_phi:
        The derivatives of all basis functions.

    :param inv_J:
        The inverse of the Jacobian matrix of the transformation.

    :return:
        The field vectors H, as well as the magnitude H_mag.
    '''

    # the number of evaluation points
    num_eval = d_phi.shape[0]

    # allocate the return array
    grad_phi = np.zeros((num_eval, 3))

    # evaluate
    grad_phi[:, 0] = d_phi[:, :, 0] @ x[c]
    grad_phi[:, 1] = d_phi[:, :, 1] @ x[c]
    grad_phi[:, 2] = d_phi[:, :, 2] @ x[c]

    # transform it to the global domain
    for m in range(num_eval):
        grad_phi[m, :] = inv_J[m, :, :].T @ grad_phi[m, :]


    # return also the magnitude
    mag = np.linalg.norm(grad_phi, axis=1)

    return grad_phi, mag

def evaluate_laplace_solution(positions, gmsh_model, vol_tags, solution):
    '''Compute the B field (interior or exterior) at certain posisions.

    :param positions:
        A numpy array of size (M x 3) with the coordinates in the columns.

    :param gmsh_model:
        The gmsh model object.

    :param vol_tags:
        The gmsh volume tags.
            
    :param solution:
        A solution vector.

    :return:
        A numpy matrix of dimension (M x 3) with the B field components in the
        columns. 
    '''

    # the number of positions
    num_pos = positions.shape[0]

    # The nodes are not sorted correctly. I don't know why...
    # But we need to get them like this:
    node_tags, _, _ = gmsh_model.mesh.getNodes()
    num_nodes = len(node_tags)
    node_tags = np.unique(node_tags)

    # we now make an array of unique mesh nodes.
    nodes = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        nodes[i, :] = gmsh_model.mesh.getNode(node_tags[i])[0]

    # we now allocate a return vector
    field = np.zeros((num_pos, 3))

    # domain specifier for the volume number
    domain_spec = np.zeros((num_pos, ), dtype=np.int32)

    # this is a mask for the outside points
    outside_mask = np.zeros((num_pos, ), dtype=bool)

    # for some reason, gmsh renumbers nodes...
    # we need to make a mapping between gmsh node tags and my node tags
    node_map = np.zeros((num_nodes, ), dtype=np.int64)

    # loop over all nodes
    for i in range(num_nodes):
        # get the coordinates
        coord, _, _, _ = gmsh_model.mesh.get_node(i+1)
        # search for this nodes in my nodes
        diff_x = coord[0] - nodes[:, 0]
        diff_y = coord[1] - nodes[:, 1]
        diff_z = coord[2] - nodes[:, 2]
        # the distance
        dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        # get the minimum
        node_map[i] = np.argmin(dist)


    # we loop over all of the positions
    for i in range(num_pos):

        for j, v in enumerate(vol_tags):
            
            # check if this point is inside or outside
            is_inside = gmsh_model.is_inside(3, v, positions[i, :])

            if is_inside == True:
                domain_spec[i] = j + 1
                break

        if is_inside == True:
            # print('position {}'.format(positions[i, :]))
            # get the element and local coordinates
            el, type, node_tags, u, v, w = gmsh_model.mesh.get_element_by_coordinates(positions[i, 0], positions[i, 1], positions[i, 2], dim=3)

            # the node indices of my numbering
            node_indx = node_map[node_tags-1]

            # make the finite element
            finite_element = FiniteElement(type)

            # evaluate the derivatives at these points
            d_phi = finite_element.evaluate_basis_derivative(np.array([[u, v, w]]).flatten())
            # d_phi = d_phi[:, indx, :]
            # compute the Jabcobi matrix
            J = finite_element.compute_J(node_indx, nodes, d_phi)
                
            # invert the Jacobian
            inv_J = finite_element.compute_J_inv(J)

            # compute also the field vector
            field[i, :], _ = compute_grad_phi(node_indx, solution, d_phi, inv_J)

        else:

            # mark this point as outside
            outside_mask[i] = True

    return field

def evaluate_curl_curl_solution(positions, gmsh_model, vol_tags, solution, glob_ids):
    '''Evaluate the B field (interior or exterior) at certain posisions.
    This function is to use for the vector potential formulation curl curl equation.

    :param positions:
         A numpy array of size (M x 3) with the coordinates in the columns.

    :param gmsh_model:
        The gmsh model object.

    :param vol_tags:
        The gmsh volume tags.
                
    :param solution:
        A solution vector.

    :return:
        A numpy matrix of dimension (M x 3) with the B field components in the
        columns. 
    '''


    # the number of positions
    num_pos = positions.shape[0]

    print('evaluate field at {} positions...'.format(num_pos))

    # The nodes are not sorted correctly. I don't know why...
    # But we need to get them like this:
    node_tags, _, _ = gmsh_model.mesh.getNodes()
    num_nodes = len(node_tags)
    node_tags = np.unique(node_tags)

    # we now make an array of unique mesh nodes.
    nodes = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        nodes[i, :] = gmsh_model.mesh.getNode(node_tags[i])[0]

    # we now allocate a return vector
    B_ret = np.zeros((num_pos, 3))

    # domain specifier for the volume number
    domain_spec = np.zeros((num_pos, ), dtype=np.int32)

    # this is a mask for the outside points
    outside_mask = np.zeros((num_pos, ), dtype=bool)

    # for some reason, gmsh renumbers nodes...
    # we need to make a mapping between gmsh node tags and my node tags
    node_map = np.zeros((num_nodes, ), dtype=np.int64)

    # loop over all nodes
    for i in range(num_nodes):
        # get the coordinates
        coord, _, _, _ = gmsh_model.mesh.get_node(i+1)
        # search for this nodes in my nodes
        diff_x = coord[0] - nodes[:, 0]
        diff_y = coord[1] - nodes[:, 1]
        diff_z = coord[2] - nodes[:, 2]
        # the distance
        dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        # get the minimum
        node_map[i] = np.argmin(dist)


    # the number of edge basis functions per element
    num_el_dofs = 6
    

    # we loop over all of the positions
    for i in range(num_pos):

        for j, v in enumerate(vol_tags):
                
            # check if this point is inside or outside
            is_inside = gmsh_model.is_inside(3, v, positions[i, :])

            if is_inside == True:
                domain_spec[i] = j + 1
                break

            # print('is inside = {}'.format(is_inside))

        if is_inside == True:

            # print('position {}'.format(positions[i, :]))

            # get the element and local coordinates
            el, type, node_tags, u, v, w = gmsh_model.mesh.get_element_by_coordinates(positions[i, 0], positions[i, 1], positions[i, 2], dim=3)


            # print('element = {}'.format(el))            

            # the index of this element in the orientations table
            el_index = el - 1

            # the node indices of my numbering
            node_indx = node_map[node_tags-1]

            # make the finite element
            finite_element = FiniteElement(type)

            # evaluate the derivatives at these points
            d_phi = finite_element.evaluate_basis_derivative(np.array([[u, v, w]]).flatten())
            # d_phi = d_phi[:, indx, :]
            # compute the Jabcobi matrix
            J = finite_element.compute_J(node_indx, nodes, d_phi)
                    
            # invert the Jacobian
            det_J = finite_element.compute_J_det(J)

            # evaluate the curls of the edge shape functions at the evaluation points
            curls, num_orientations = finite_element.get_curl_edge_basis_functions(gmsh_model.mesh, np.array([[u, v, w]]).flatten(), type, 1)
            
            # get the global orientations of the finite elements
            orientation = gmsh_model.mesh.getBasisFunctionsOrientationForElement(el, 'CurlHcurlLegendre0') 


            # print('orientation = {}'.format(orientation))

            # get the  basis functions with this orientations
            curl_w_hat = curls[3*orientation*num_el_dofs:3*(orientation+1)*num_el_dofs]
            
            # get the keys for this entity
            _, entity_keys, _ = gmsh_model.mesh.getKeysForElement(el, 'CurlHcurlLegendre0', returnCoord=False)



            # compute all products 1/det(dF)*(dF.w)^T v(B) (dF.w)
            for k in range(num_el_dofs):
                
                # the basis function in global coordinates
                B_ret[i, :] += J[0, :, :] @ curl_w_hat[3*k:3*(k+1)] * solution[entity_keys[k] - 1] / det_J[0]

        else:

            # mark this point as outside
            outside_mask[i] = True

    return B_ret

def evaluate_interpolation(positions, gmsh_model, vol_tags, nodal_values):
    '''Compute a FEM interpolation at certain posisions.

    :param positions:
        A numpy array of size (M x 3) with the coordinates in the columns.

    :param gmsh_model:
        The gmsh model object.

    :param vol_tags:
        The gmsh volume tags.
            
    :param nodal_values:
        A list of nodal values vectors. There should be one vector for each domain.

    :return:
        A numpy matrix of dimension (M x N) where M is the number of evaluation
        positions and N is the number of coordinates of the nodal values. 
    '''

    # the number of positions
    num_pos = positions.shape[0]

    # the number of coordinates
    if len(nodal_values[0].shape) == 1:
        num_coords = 1
    else:
        num_coords = nodal_values[0].shape[1]

    # The nodes are not sorted correctly. I don't know why...
    # But we need to get them like this:
    node_tags, _, _ = gmsh_model.mesh.getNodes()
    num_nodes = len(node_tags)
    node_tags = np.unique(node_tags)

    # we now make an array of unique mesh nodes.
    nodes = np.zeros((num_nodes, 3))

    for i in range(num_nodes):
        nodes[i, :] = gmsh_model.mesh.getNode(node_tags[i])[0]

    # we now allocate a return vector
    field = np.zeros((num_pos, num_coords))

    # this is a mask for the outside points
    outside_mask = np.zeros((num_pos, ), dtype=bool)

    # for some reason, gmsh renumbers nodes...
    # we need to make a mapping between gmsh node tags and my node tags
    node_map = np.zeros((num_nodes, ), dtype=np.int64)

    # loop over all nodes
    for i in range(num_nodes):
        # get the coordinates
        coord, _, _, _ = gmsh_model.mesh.get_node(i+1)
        # search for this nodes in my nodes
        diff_x = coord[0] - nodes[:, 0]
        diff_y = coord[1] - nodes[:, 1]
        diff_z = coord[2] - nodes[:, 2]
        # the distance
        dist = np.sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z)
        # get the minimum
        node_map[i] = np.argmin(dist)

    # this variable will be used as a domain specifyer
    domain_spec = -1

    # we loop over all of the positions
    for i in range(num_pos):

        for j, v in enumerate(vol_tags):

            # check if this point is inside or outside
            is_inside = gmsh_model.is_inside(3, v, positions[i, :])

            if is_inside:
                domain_spec = j
                break

        if is_inside:

            # get the element and local coordinates
            _, el_type, \
            node_tags, \
            u, v, w = gmsh_model.mesh.get_element_by_coordinates(positions[i, 0],
                                                                 positions[i, 1],
                                                                 positions[i, 2], dim=3)

            # the node indices of my numbering
            node_indx = node_map[node_tags-1]

            # make the finite element
            finite_element = FiniteElement(el_type)

            # evaluate the derivatives at these points
            phi = finite_element.evaluate_basis(np.array([[u, v, w]]).flatten())

            # compute the field
            if num_coords == 1:
                field[i] = phi @ nodal_values[domain_spec][node_indx]
            else:
                for j in range(num_coords):
                    field[i, j] = phi @ nodal_values[domain_spec][node_indx, j]

        else:

            # mark this point as outside
            outside_mask[i] = True

    return field

def evaluate_fem_solution_node_interpolation(gmsh_model, points, solver, coil_list, x, domain_list):
    '''Evaluate a FEM solution by nodal interpolation.

    :param gmsh_model:
        The gmsh model. This stores the information of the mesh and its entities.

    :param points:
        The evaluation points.

    :param solver:
        The solver.

    :param coil_list:
        The list of coils.

    :param x:
        The solution vector.

    :param domain inices.
        A list of domains to interpolate the solution in.

    :return:
        The B fields at the evaluation points.
    '''

    # these are the node positions
    nodes = solver.curl_curl_factory.nodes

    # the coil fields at the nodes
    b_coil = 0.0*nodes
    for coil in coil_list:
        b_coil += coil.compute_B(nodes)

    # the B fields at the nodes of the mesh one element for each material
    b_node_list = []

    # loop over the domains
    for indx in domain_list:
        # lets average the element fields to get the nodal fields
        b_node_list.append(solver.curl_curl_factory.compute_nodal_fields(x, [indx-1]) + b_coil)

    return evaluate_interpolation(points, gmsh_model, domain_list, b_node_list)